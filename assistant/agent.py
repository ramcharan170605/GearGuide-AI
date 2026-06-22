"""
Agent loop and conversation management for Dealer Assistant

This module now implements the LLM-FIRST architecture where the LLM is the primary
reasoning engine for:
- Understanding user intent
- Deciding which tools to call
- Managing conversation context
- Handling clarification
- Ensuring grounding
- Providing guardrails

The LLM drives the entire reasoning pipeline:
User Query ↓
LLM Understanding ↓
Tool Decision ↓
Retrieval Decision ↓
Tool Execution / Retrieval ↓
Context Assembly ↓
LLM Response Generation ↓
Final Response

Fallback: If LLM is unavailable, falls back to pattern-based logic

Requirements: LLM-FIRST architecture as per assignment.md
"""

import re
import os
from typing import Optional, Any
from dataclasses import dataclass, field
import json

# Import the LLM provider
from assistant.llm_provider import Message, get_llm_provider, LLMConfig, LLMReturn
from assistant.retrieval import CatalogueRetriever
from assistant.tools import ToolRegistry, CheckStockResult, CreateOrderResult, OrderItem


@dataclass
class ConversationContext:
    """
    Maintains conversation state across turns for LLM-driven reasoning.

    The LLM uses this context to understand follow-up queries without
    repeated information (e.g., "What is the stock of BRK-1007?" followed by
    "Can I order 5?" should understand that 5 refers to BRK-1007).

    Requirements: CONV-001, CONV-002 (multi-turn dialogue)
    """
    history: list[Message] = field(default_factory=list)
    current_query: Optional[str] = None
    last_tool_result: Optional[dict] = None
    last_retrieval_results: Optional[list[dict]] = None
    # Track entities mentioned in conversation for follow-up reasoning
    mentioned_skus: list[str] = field(default_factory=list)
    mentioned_vehicles: list[dict] = field(default_factory=list)
    pending_clarification: Optional[str] = None
    conversation_metadata: dict = field(default_factory=dict)

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        self.history.append(Message(role=role, content=content))

    def extract_entities_from_query(self, query: str):
        """Extract SKUs and vehicle references from a query."""
        # Extract SKUs
        skus = re.findall(r"\b[A-Z]{2,4}-\d{3,5}\b", query)
        self.mentioned_skus.extend(skus)

    def clear(self):
        """Clear the conversation context."""
        self.history = []
        self.current_query = None
        self.last_tool_result = None
        self.last_retrieval_results = None
        self.mentioned_skus = []
        self.mentioned_vehicles = []
        self.pending_clarification = None
        self.conversation_metadata = {}


@dataclass
class ToolDescription:
    """Structured description of a tool for LLM consumption."""
    name: str
    description: str
    parameters: dict
    required: list[str]


class DealerAssistant:
    """
    LLM-Driven Dealer Assistant.

    The LLM is the primary reasoning engine that:
    1. Understands user queries (intent, entities, scope)
    2. Decides whether to use tools or retrieval
    3. Selects which tool to call and with what parameters
    4. Determines when clarification is needed
    5. Manages conversation context for multi-turn dialogue
    6. Ensures responses are grounded in retrieved data
    7. Handles guardrails (off-topic, unsafe queries)

    Architecture:
        User Query
            ↓
        LLM Understanding (intent detection, scope checking)
            ↓
        Tool/Retrieval Decision (LLM decides what to do)
            ↓
        Tool Execution / Retrieval
            ↓
        Context Assembly
            ↓
        LLM Response Generation (grounded in context)
            ↓
        Final Response

    Requirements: LLM-FIRST architecture, CONV-001 to CONV-004,
                  TOOL-004, TOOL-006, GROUND-001, GROUND-002
    """

    def __init__(
        self,
        retriever: CatalogueRetriever,
        tool_registry: ToolRegistry,
        config: Optional[LLMConfig] = None
    ):
        self.retriever = retriever
        self.tools = tool_registry
        self.config = config or LLMConfig()
        self.context = ConversationContext()
        self.confidence_threshold: float = 0.7

        # LLM provider
        self.provider = get_llm_provider(self.config)
        self._llm_available = self.provider.is_available()

        # Build tool descriptions for LLM
        self._tool_descriptions = self._build_tool_descriptions()

        # System prompt for the LLM
        self._system_prompt = self._build_system_prompt()

    def _build_tool_descriptions(self) -> list[ToolDescription]:
        """Build structured tool descriptions for LLM consumption."""
        descriptions = []

        # Get tool info from registry
        tool_names = self.tools.list_tools()
        for name in tool_names:
            info = self.tools.get_tool_info(name)
            descriptions.append(ToolDescription(
                name=name,
                description=info.get("description", ""),
                parameters=info.get("schema", {}).get("properties", {}),
                required=info.get("schema", {}).get("required", [])
            ))

        return descriptions

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the LLM."""
        # Format tool descriptions
        tool_desc_text = "\n".join(
            f"- {td.name}: {td.description}"
            for td in self._tool_descriptions
        )

        system_prompt = f"""You are GearGuide-AI, a helpful dealer assistant for auto parts.

Your capabilities:
- Check stock availability for products by SKU
- Find parts that fit specific vehicles
- Create orders for dealers
- Search the product catalogue
- Answer questions about products

Available tools:
{tool_desc_text}

Guidelines:
1. ALWAYS ground your responses in the provided context (retrieval results, tool outputs)
2. If you mention a product SKU, price, or stock, it MUST come from the context
3. If the user's query is ambiguous, ask for clarification
4. If the query is off-topic (not about auto parts, stock, or orders), politely decline
5. For follow-up queries, use the conversation context to understand references
6. When calling tools, use the exact parameter names and types from the tool descriptions
7. If you don't have enough information to answer confidently, ask for clarification
8. Be concise and direct in your responses

Response format:
- For direct answers: Provide the answer with relevant details
- For clarification: Ask a specific question about what's missing
- For tool calls: Return JSON with tool name and parameters, e.g.: {{ "tool": "check_stock", "parameters": {{ "sku": "BRK-1007" }} }}
- For off-topic: Politely decline and explain your domain

Remember: You are ONLY for auto parts, stock, and order assistance. Stay on topic.

Previous conversation context:
{{previous_context}}"""

        return system_prompt

    def _format_tool_descriptions_for_llm(self) -> str:
        """Format tool descriptions in a way the LLM can understand."""
        tool_xml = "<tools>\n"
        for td in self._tool_descriptions:
            tool_xml += f"  <tool name='{td.name}'>\n"
            tool_xml += f"    <description>{td.description}</description>\n"
            tool_xml += f"    <parameters>\n"
            for param_name, param_info in td.parameters.items():
                param_type = param_info.get("type", "string")
                param_desc = param_info.get("description", "")
                required = param_name in td.required
                tool_xml += f"      <parameter name='{param_name}' type='{param_type}' required='{required}'>\n"
                tool_xml += f"        {param_desc}\n"
                tool_xml += f"      </parameter>\n"
            tool_xml += f"    </parameters>\n"
            tool_xml += f"  </tool>\n"
        tool_xml += "</tools>"
        return tool_xml

    def _build_llm_messages(
        self,
        query: str,
        retrieval_results: Optional[list[dict]] = None,
        tool_results: Optional[list[dict]] = None
    ) -> list[Message]:
        """Build the message list for the LLM with full context."""
        messages = []

        # Build system prompt with context
        conversation_context = self.context.get_context_summary(2000) if hasattr(self.context, 'get_context_summary') else ""
        actual_system_prompt = self._system_prompt.replace("{{previous_context}}", conversation_context)

        # System message
        messages.append(Message(role="system", content=actual_system_prompt))

        # Tool descriptions (as a system message)
        tool_desc = self._format_tool_descriptions_for_llm()
        messages.append(Message(role="system", content=f"Available tools:\n{tool_desc}"))

        # Conversation history (skip the first system message)
        for msg in self.context.history:
            messages.append(msg)

        return messages

    def _format_retrieval_results(self, results: list[dict]) -> str:
        """Format retrieval results for LLM consumption."""
        if not results:
            return "No retrieval results."

        lines = []
        for i, product in enumerate(results[:10], 1):
            sku = product.get("sku", "N/A")
            name = product.get("name", "N/A")
            category = product.get("category", "N/A")
            brand = product.get("brand", "N/A")
            vehicle_fitment = product.get("vehicle_fitment", "N/A")
            price = product.get("price_inr", 0)
            stock = product.get("stock", 0)
            description = product.get("description", "")
            similarity = product.get("similarity_score", 0)

            lines.append(f"Result {i}:")
            lines.append(f"  SKU: {sku}")
            lines.append(f"  Name: {name}")
            lines.append(f"  Category: {category}")
            lines.append(f"  Brand: {brand}")
            lines.append(f"  Vehicle Fitment: {vehicle_fitment}")
            lines.append(f"  Price: ₹{price}")
            lines.append(f"  Stock: {stock}")
            lines.append(f"  Description: {description}")
            lines.append(f"  Similarity: {similarity:.4f}")
            lines.append("")

        return "\n".join(lines)

    def _format_tool_results(self, results: list[dict]) -> str:
        """Format tool results for LLM consumption."""
        if not results:
            return "No tool results."

        lines = []
        for result in results:
            if isinstance(result, CheckStockResult):
                lines.append(f"check_stock result:")
                lines.append(f"  SKU: {result.sku}")
                lines.append(f"  Name: {result.name}")
                lines.append(f"  Stock: {result.stock}")
                lines.append(f"  In Stock: {result.in_stock}")
                lines.append(f"  Price: ₹{result.price_inr}")
                lines.append(f"  Category: {result.category}")
            elif isinstance(result, CreateOrderResult):
                lines.append(f"create_order result:")
                lines.append(f"  Order ID: {result.order_id}")
                lines.append(f"  Dealer: {result.dealer}")
                lines.append(f"  Total: ₹{result.total_inr}")
                lines.append(f"  Status: {result.status}")
                lines.append(f"  Items: {len(result.items)}")
                for item in result.items:
                    lines.append(f"    - {item.sku}: {item.quantity}")
            else:
                lines.append(f"Tool result: {json.dumps(result, indent=2)}")

        return "\n".join(lines)

    def _extract_sku(self, query: str) -> Optional[str]:
        """Extract SKU from query."""
        match = re.search(r"\b[A-Z]{2,4}-\d{3,5}\b", query)
        return match.group(0) if match else None

    def _parse_tool_call_from_response(self, response: str) -> Optional[dict]:
        """
        Parse a tool call from the LLM's response.

        The LLM is instructed to return JSON for tool calls.
        """
        # Try to parse JSON from the response
        try:
            # Look for JSON in the response
            # The LLM might wrap it in markdown or other text
            response_stripped = response.strip()

            # Try direct JSON parse
            if response_stripped.startswith('{') and response_stripped.endswith('}'):
                data = json.loads(response_stripped)
                if "tool" in data and "parameters" in data:
                    return data
                elif "function_call" in data:
                    # Handle function call format
                    func_call = data["function_call"]
                    return {
                        "tool": func_call.get("name"),
                        "parameters": func_call.get("arguments", {})
                    }
        except (json.JSONDecodeError, AttributeError):
            pass

        # Pattern 2: "Call: check_stock with sku=BRK-1007"
        pattern1 = r"Call:\s*(\w+)\s+with\s+(.+)"
        match = re.search(pattern1, response, re.IGNORECASE)
        if match:
            tool_name = match.group(1)
            params_str = match.group(2)
            return self._parse_params(params_str, tool_name)

        # Pattern 3: "Use tool: check_stock"
        pattern3 = r"Use\s+tool:\s*(\w+)"
        match = re.search(pattern3, response, re.IGNORECASE)
        if match:
            tool_name = match.group(1)
            return {"tool": tool_name, "parameters": {}}

        return None

    def _parse_params(self, params_str: str, tool_name: str) -> dict:
        """Parse parameters from a comma-separated key=value string."""
        params = {}

        # Split by comma
        parts = [p.strip() for p in params_str.split(",")]

        for part in parts:
            if "=" in part:
                key, value = part.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"\'')

                # Try to infer type based on the tool's expected parameters
                tool_info = self.tools.get_tool_info(tool_name)
                if tool_info and "schema" in tool_info:
                    param_schema = tool_info["schema"].get("properties", {}).get(key, {})
                    param_type = param_schema.get("type", "string")

                    # Convert value to appropriate type
                    if param_type == "integer":
                        try:
                            value = int(value)
                        except ValueError:
                            pass
                    elif param_type == "number":
                        try:
                            value = float(value)
                        except ValueError:
                            pass
                    elif param_type == "boolean":
                        value = value.lower() in ["true", "yes", "1"]

                params[key] = value

        return {"tool": tool_name, "parameters": params}

    def _execute_tool(self, tool_name: str, parameters: dict) -> Any:
        """Execute a tool with the given parameters."""
        tool = self.tools.get_tool(tool_name)
        if tool is None:
            raise ValueError(f"Tool {tool_name} not found")

        return tool(**parameters)

    def _format_response_from_tool(self, tool_name: str, tool_result: Any) -> str:
        """Format a tool result into a user-friendly response."""
        if tool_name == "check_stock":
            result = tool_result
            status = "In Stock" if result.in_stock else "Out of Stock"
            return f"{result.name} (SKU: {result.sku}) - ₹{result.price_inr}, Stock: {result.stock}, Status: {status}"

        elif tool_name == "create_order":
            result = tool_result
            items = "\n".join(f" - {i.sku}: {i.quantity}" for i in result.items)
            return f"Order {result.order_id} for {result.dealer} - Total: ₹{result.total_inr}\nItems:\n{items}"

        elif tool_name == "find_parts_by_vehicle":
            results = tool_result
            if not results:
                return "No parts found for that vehicle."
            parts = []
            for i, p in enumerate(results[:10], 1):
                parts.append(f"{i}. {p.get('name','N/A')} (SKU: {p.get('sku','N/A')}) - ₹{p.get('price_inr',0)}, Stock: {p.get('stock',0)}")
            return "Matching products:\n" + "\n".join(parts)

        else:
            return str(tool_result)

    def _is_off_topic_lightweight(self, query: str) -> bool:
        """
        Lightweight off-topic detection.

        This is used for efficiency. The LLM can be used for ambiguous cases.
        """
        query_lower = query.lower()

        # Definitely off-topic phrases
        off_topic_phrases = [
            'stock market', 'market', 'weather', 'time', 'date', 'today',
            'joke', 'funny', 'laugh', 'cricket', 'football', 'sports',
            'news', 'politics', 'election', 'president', 'prime minister',
            'movie', 'film', 'song', 'music', 'concert',
            'food', 'restaurant', 'recipe', 'cook',
            'health', 'doctor', 'hospital', 'medicine',
            'school', 'college', 'university', 'education',
            'love', 'relationship', 'dating',
            'god', 'religion', 'prayer'
        ]

        for phrase in off_topic_phrases:
            if phrase in query_lower:
                return True

        # Definitely on-topic keywords
        on_topic_keywords = [
            'part', 'parts', 'product', 'products', 'item', 'items',
            'stock', 'order', 'price', 'inventory', 'available', 'availability',
            'brake', 'tyre', 'tire', 'chain', 'oil', 'engine', 'filter',
            'bike', 'motorcycle', 'scooter', 'car', 'vehicle', 'auto',
            'bajaj', 'ktm', 'royal enfield', 'honda', 'yamaha', 'tvs', 'hero', 'suzuki',
            'sku', 'catalogue', 'catalog', 'purchase', 'buy', 'sell',
            'lube', 'lubricant', 'pad', 'disc', 'rotor', 'sprocket',
            'seat', 'cover', 'mirror', 'light', 'horn', 'battery',
            'exhaust', 'muffler', 'guard', 'assembly',
            'show', 'have', 'cheapest', 'cheap', 'expensive', 'cost'
        ]

        # If it contains on-topic keywords, it's probably on-topic
        if any(keyword in query_lower for keyword in on_topic_keywords):
            return False

        return False

    def _check_guardrails_llm(self, query: str) -> tuple[bool, str]:
        """
        Check if a query is off-topic or unsafe using LLM reasoning.

        For efficiency, we first use a lightweight check.
        """
        # Lightweight check first
        if self._is_off_topic_lightweight(query):
            return True, self._generate_guardrail_response()

        # If LLM is available and lightweight check is uncertain, use LLM
        if self._llm_available:
            try:
                messages = [
                    Message(
                        role="system",
                        content="You are a guardrail for an auto parts assistant. Determine if the query is on-topic (about auto parts, stock, orders) or off-topic. Respond with ONLY 'on-topic' or 'off-topic'."
                    ),
                    Message(role="user", content=f"Query: {query}")
                ]
                result = self.provider.chat(messages)
                if "off-topic" in result.content.lower():
                    return True, self._generate_guardrail_response()
            except Exception:
                pass

        return False, ""

    def _generate_guardrail_response(self) -> str:
        """Generate a polite guardrail response."""
        return "I'm sorry, I can only help with auto parts, stock, and orders. Is there something related to auto parts I can help you with?"

    def _needs_clarification_llm(self, query: str) -> tuple[bool, str]:
        """
        Determine if clarification is needed using LLM reasoning.

        This replaces the pattern-based clarification detection.
        """
        # Skip clarification for direct SKU queries
        if self._extract_sku(query):
            return False, ""

        # Check for obvious action queries with enough info
        action_keywords = [
            "stock of ", "check stock", "what's the stock", "what is the stock",
            "price of ", "what is the price", "what's the price",
            "availability", "how many", "units of ", "units available", "in stock"
        ]

        query_lower = query.lower()
        for keyword in action_keywords:
            if keyword in query_lower:
                # Has SKU or vehicle info
                if self._extract_sku(query) or " for " in query_lower:
                    return False, ""

        # Check for order queries
        order_keywords = ["place order", "create order", "order for", "i want to order"]
        for keyword in order_keywords:
            if keyword in query_lower:
                # Has enough info (dealer and SKU)
                if self._extract_sku(query):
                    return False, ""

        # Short queries likely need clarification
        if len(query.split()) <= 2:
            # Unless they're specific enough
            specific_keywords = ['stock', 'order', 'price', 'find', 'search', 'check']
            if not any(w in query_lower for w in specific_keywords):
                return True, "Could you please provide more details?"

        # Ambiguous queries that need vehicle specification
        ambiguous_patterns = [
            r"i need tyres\b", r"tyres for", r"i need brakes\b",
            r"chain lube\b", r"engine oil\b", r"show me\b",
            r"what (?:parts|items|products)\b", r"give me\b"
        ]

        for pattern in ambiguous_patterns:
            if re.search(pattern, query_lower):
                return True, "Which vehicle? Please specify make and model (e.g., 'Bajaj Pulsar 150')."

        # Use LLM for ambiguous cases if available
        if self._llm_available:
            try:
                messages = [
                    Message(
                        role="system",
                        content="Determine if the query is clear enough to answer or needs clarification. Respond with ONLY 'clear' or 'needs_clarification'. If unclear, also suggest what information is missing."
                    ),
                    Message(role="user", content=f"Query: {query}")
                ]
                result = self.provider.chat(messages)
                if "needs_clarification" in result.content.lower():
                    return True, "Could you please provide more details?"
            except Exception:
                pass

        return False, ""

    def _generate_clarification(self, query: str) -> str:
        """Generate a clarification question."""
        query_lower = query.lower()
        if any(w in query_lower for w in ['tyre', 'brake', 'chain', 'oil', 'parts']):
            return "Which vehicle? Please specify make and model (e.g., 'Bajaj Pulsar 150')."
        return "Could you please provide more details?"

    def _generate_retrieval_response(self, query: str, retrieval_results: list[dict]) -> str:
        """Generate response from retrieval."""
        if not retrieval_results:
            return "No products found. Try different keywords."
        parts = []
        for i, p in enumerate(retrieval_results[:5], 1):
            parts.append(f"{i}. {p.get('name','N/A')} (SKU: {p.get('sku','N/A')}) - ₹{p.get('price_inr',0)}, Stock: {p.get('stock',0)}")
        return "Matching products:\n" + "\n".join(parts)

    def _format_order_result(self, result: CreateOrderResult) -> str:
        """Format order result."""
        items = "\n".join(f" - {i.sku}: {i.quantity}" for i in result.items)
        return f"Order {result.order_id} for {result.dealer} - Total: ₹{result.total_inr}\nItems:\n{items}"

    def _format_stock_result(self, result: CheckStockResult) -> str:
        """Format stock result."""
        status = "In Stock" if result.in_stock else "Out of Stock"
        return f"{result.name} (SKU: {result.sku}) - ₹{result.price_inr}, Stock: {result.stock}, Status: {status}"

    def _check_grounding(self, response: str, retrieval_results: list[dict]) -> bool:
        """Check grounding."""
        retrieved_skus = {p.get('sku', '') for p in retrieval_results}
        mentioned_skus = re.findall(r"\b[A-Z]{2,4}-\d{3,5}\b", response)
        return all(sku in retrieved_skus for sku in mentioned_skus)

    def _parse_order_query(self, query: str):
        """Parse order query (fallback method)."""
        dealer = None
        items = []
        match = re.search(r"for\s+([A-Za-z\s]+)$", query)
        if match:
            dealer = match.group(1).strip()
        sku = self._extract_sku(query)
        if sku:
            qty_match = re.search(r"(\d+)\s+units?\s+of\s+", query, re.IGNORECASE)
            quantity = int(qty_match.group(1)) if qty_match else 1
            items.append(OrderItem(sku=sku, quantity=quantity))
        return dealer, items if items else (None, None)

    def _parse_vehicle_query(self, query: str):
        """Parse vehicle query (fallback method)."""
        query_lower = query.lower()
        if " for " in query_lower:
            parts = query_lower.split(" for ")
            if len(parts) >= 2:
                vehicle_part = parts[-1]
                # Split vehicle part into make and model
                words = vehicle_part.split()
                if len(words) >= 2:
                    make = words[0].title()
                    model = " ".join(words[1:]).title()
                    return make, model, None
        return None, None, None

    def _execute_tool_action(self, query: str, retrieval_results: list[dict]) -> str:
        """Execute tool action (fallback method)."""
        query_lower = query.lower()
        try:
            # create_order: check first (higher priority)
            if "order" in query_lower:
                dealer, items = self._parse_order_query(query)
                if dealer and items:
                    tool = self.tools.get_tool("create_order")
                    if tool:
                        result = tool(dealer, items)
                        if isinstance(result, CreateOrderResult):
                            return self._format_order_result(result)

            # check_stock: handle stock, availability, and price queries
            if any(w in query_lower for w in ["stock", "availability", "price", "how many"]):
                sku = self._extract_sku(query)
                if sku:
                    tool = self.tools.get_tool("check_stock")
                    if tool:
                        result = tool(sku)
                        if isinstance(result, CheckStockResult):
                            return self._format_stock_result(result)

            # find_parts_by_vehicle: handle vehicle queries
            if " for " in query_lower:
                make, model, year = self._parse_vehicle_query(query)
                if make and model:
                    tool = self.tools.get_tool("find_parts_by_vehicle")
                    if tool:
                        result = tool(make, model, year)
                        return self._format_response_from_tool("find_parts_by_vehicle", result)

            return self._generate_retrieval_response(query, retrieval_results)
        except Exception as e:
            return f"Error: {str(e)}. Please try again."

    def _decide_action(self, query: str, retrieval_results: list[dict]) -> str:
        """Decide action type (fallback method)."""
        query_lower = query.lower()
        # Check for stock/price queries
        if any(p in query_lower for p in ["stock of ", "check stock", "what's the stock", "what is the stock",
                "what is the price", "what's the price", "price of ",
                "availability", "how many", "units of ", "units available", "in stock"]):
            return "tool"
        # Check for order queries
        if any(p in query_lower for p in ["place order", "create order", "make an order", "order for", "i want to order"]):
            return "tool"
        return "retrieval"

    def process_query(self, query: str) -> str:
        """
        Process a user query and return a response.

        This is the main entry point. The flow is:
        1. Check guardrails (off-topic detection)
        2. Check for pending clarification
        3. Check if clarification is needed
        4. Perform retrieval
        5. Decide action (LLM-driven or fallback)
        6. Execute tool or generate retrieval response
        7. Validate grounding
        8. Return response

        Requirements: CONV-001, CONV-002, CONV-003, TOOL-004, GROUND-001, GROUND-002
        """
        self.context.current_query = query
        self.context.add_message("user", query)

        # Extract entities from query
        self.context.extract_entities_from_query(query)

        # Check guardrails first (LLM-based or lightweight)
        is_off_topic, guardrail_response = self._check_guardrails_llm(query)
        if is_off_topic:
            self.context.add_message("assistant", guardrail_response)
            return guardrail_response

        # Check if we're waiting for clarification
        if self.context.pending_clarification:
            self.context.pending_clarification = None
            # Process the clarification response as a new query
            return self.process_query(guardrail_response)

        # Check if clarification is needed (LLM-driven)
        needs_clarification, clarification_msg = self._needs_clarification_llm(query)
        if needs_clarification:
            self.context.pending_clarification = clarification_msg
            self.context.add_message("assistant", clarification_msg)
            return clarification_msg

        # Perform retrieval
        retrieval_results = self.retriever.search(query, k=5)
        self.context.last_retrieval_results = retrieval_results

        # Check confidence - but if we have a SKU in the query, trust it
        has_sku = self._extract_sku(query) is not None
        if (retrieval_results and
            retrieval_results[0].get('similarity_score', 0) < self.confidence_threshold and
            not has_sku):
            clarification = self._generate_clarification(query)
            self.context.pending_clarification = clarification
            self.context.add_message("assistant", clarification)
            return clarification

        # LLM is mandatory - use LLM-driven approach
        if not self._llm_available:
            raise RuntimeError(
                "LLM provider is required but not available. "
                "Please configure GEMINI_API_KEY or OPENAI_API_KEY in your .env file."
            )

        try:
            return self._process_with_llm(query, retrieval_results)
        except Exception as e:
            # LLM processing failed - this is a hard error
            raise RuntimeError(f"LLM processing failed: {e}")

    def _process_with_llm(self, query: str, retrieval_results: list[dict]) -> str:
        """
        Process query using LLM-driven reasoning.
        """
        # Build LLM messages with full context
        messages = self._build_llm_messages(query, retrieval_results)

        # Add retrieval context to the last message
        if retrieval_results:
            retrieval_context = self._format_retrieval_results(retrieval_results)
            # Add as a system message
            messages.append(Message(role="system", content=f"RETRIEVAL CONTEXT:\n{retrieval_context}"))

        # Call the LLM
        result = self.provider.chat(messages)
        llm_response = result.content

        # Parse the LLM's response for tool calls
        tool_call = self._parse_tool_call_from_response(llm_response)

        if tool_call:
            # Execute the tool
            try:
                tool_name = tool_call.get("tool")
                parameters = tool_call.get("parameters", {})

                # Get the tool and call it
                tool_result = self._execute_tool(tool_name, parameters)
                self.context.last_tool_result = {"tool": tool_name, "result": tool_result}

                # Format the tool result for the user
                response = self._format_response_from_tool(tool_name, tool_result)
                self.context.add_message("assistant", response)
                return response

            except Exception as e:
                return f"Error executing tool: {str(e)}"
        else:
            # The LLM provided a direct response
            # Validate grounding
            if not self._check_grounding(llm_response, retrieval_results):
                # If not grounded, try again with a grounding prompt
                grounding_messages = messages + [
                    Message(
                        role="system",
                        content="Your previous response was not grounded in the retrieval results. Please provide a response that only mentions products, prices, and stock from the RETRIEVAL CONTEXT above."
                    )
                ]
                try:
                    result = self.provider.chat(grounding_messages)
                    llm_response = result.content
                except Exception:
                    # Fallback to retrieval-based response
                    llm_response = self._generate_retrieval_response(query, retrieval_results)

            self.context.add_message("assistant", llm_response)
            return llm_response

    def _process_with_patterns(self, query: str, retrieval_results: list[dict]) -> str:
        """
        Process query using pattern-based logic (fallback).
        """
        # Decide and execute action
        action = self._decide_action(query, retrieval_results)
        if action == "tool":
            response = self._execute_tool_action(query, retrieval_results)
            # Tool responses are already grounded (they use catalogue data directly)
            # Skip grounding check for tool-based responses
        else:
            response = self._generate_retrieval_response(query, retrieval_results)
            # Validate grounding for retrieval-based responses
            if not self._check_grounding(response, retrieval_results):
                response = self._generate_retrieval_response(query, retrieval_results)

        self.context.add_message("assistant", response)
        return response


if __name__ == "__main__":
    # Test the agent
    print("Testing Dealer Assistant...")
    from assistant.retrieval import CatalogueRetriever
    from assistant.tools import create_default_tool_registry
    import pandas as pd

    # Initialize
    catalogue = pd.read_csv('catalogue.csv')
    retriever = CatalogueRetriever('catalogue.csv')
    retriever.load_catalogue()
    retriever.initialize_embedding_model()
    retriever.build_vector_store()
    tool_registry = create_default_tool_registry(catalogue)
    agent = DealerAssistant(retriever, tool_registry)

    # Test queries
    test_queries = [
        "What is the stock of BRK-1007?",
        "Find parts for Bajaj Pulsar 150",
        "Place an order for 5 units of BRK-1007 for ABC Motors",
        "I need tyres",
        "What's the weather today?"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        response = agent.process_query(query)
        print(f"Response: {response[:200]}...")
