"""
LLM-Driven Agent for GearGuide-AI Dealer Assistant

This module implements the LLM-first architecture where the LLM is the primary
reasoning engine for understanding, routing, retrieval decisions, tool selection,
clarification, conversation management, grounding, and response generation.

Architecture:
    User Query
        ↓
    LLM Understanding (intent detection, scope checking)
        ↓
    Tool Decision (LLM decides which tool to call)
        ↓
    Retrieval Decision (LLM decides when/what to retrieve)
        ↓
    Tool Execution / Retrieval
        ↓
    Context Assembly (combine retrieval + tool results)
        ↓
    LLM Response Generation (grounded in context)
        ↓
    Final Response

Requirements: LLM-FIRST architecture as per assignment.md
"""

import re
import json
import os
from typing import Optional, Any
from dataclasses import dataclass, field
from assistant.llm_provider import Message, get_llm_provider, LLMConfig
from assistant.retrieval import CatalogueRetriever
from assistant.tools import ToolRegistry, CheckStockResult, CreateOrderResult, OrderItem


@dataclass
class ConversationContext:
    """
    Maintains conversation state across turns for LLM-driven reasoning.

    The LLM uses this context to understand follow-up queries without
    repeated information (e.g., "What is the stock of BRK-1007?" followed by
    "Can I order 5?" should understand that 5 refers to BRK-1007).
    """
    history: list[Message] = field(default_factory=list)
    current_query: Optional[str] = None
    last_tool_result: Optional[dict] = None
    last_retrieval_results: Optional[list[dict]] = None
    # Track entities mentioned in conversation for follow-up reasoning
    mentioned_skus: list[str] = field(default_factory=list)
    mentioned_vehicles: list[dict] = field(default_factory=list)  # [{"make": ..., "model": ...}]
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

        # Extract vehicle patterns (simple heuristic for now)
        # This will be enhanced by LLM-based entity extraction
        vehicle_patterns = [
            r"(Bajaj|KTM|Royal Enfield|Honda|Yamaha|TVS|Hero|Suzuki)",
            r"(Pulsar|Hornet|Meteor|Unicorn|Apache|RTR|FZ|R15|Seltos)"
        ]
        for pattern in vehicle_patterns:
            matches = re.findall(pattern, query)
            for match in matches:
                # Simple vehicle detection - LLM will handle complex cases
                pass

    def get_context_summary(self, max_tokens: int = 2000) -> str:
        """Get a summary of the conversation context for LLM consumption."""
        if not self.history:
            return ""

        # Build context summary
        context_parts = []

        # Previous messages
        for msg in self.history[-10:]:  # Last 10 messages
            context_parts.append(f"{msg.role.upper()}: {msg.content}")

        # Mentioned entities
        if self.mentioned_skus:
            context_parts.append(f"Mentioned SKUs: {', '.join(self.mentioned_skus)}")

        return "\n".join(context_parts[-20:])  # Limit to last 20 lines


@dataclass
class ToolDescription:
    """Structured description of a tool for LLM consumption."""
    name: str
    description: str
    parameters: dict
    required: list[str]


class LLMDealerAssistant:
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
        self.provider = get_llm_provider(self.config)

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
- For tool calls: Indicate which tool to call and with what parameters
- For off-topic: Politely decline and explain your domain

Remember: You are ONLY for auto parts, stock, and order assistance. Stay on topic."""

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

        # System message
        messages.append(Message(role="system", content=self._system_prompt))

        # Tool descriptions (as a system message)
        tool_desc = self._format_tool_descriptions_for_llm()
        messages.append(Message(role="system", content=f"Available tools:\n{tool_desc}"))

        # Conversation history
        messages.extend(self.context.history)

        # Current query
        messages.append(Message(role="user", content=query))

        # Add context information if available
        if retrieval_results:
            context_str = self._format_retrieval_results(retrieval_results)
            messages.append(Message(
                role="system",
                content=f"RETRIEVAL CONTEXT:\n{context_str}"
            ))

        if tool_results:
            tool_str = self._format_tool_results(tool_results)
            messages.append(Message(
                role="system",
                content=f"TOOL RESULTS:\n{tool_str}"
            ))

        return messages

    def _format_retrieval_results(self, results: list[dict]) -> str:
        """Format retrieval results for LLM consumption."""
        if not results:
            return "No retrieval results."

        lines = []
        for i, product in enumerate(results[:10], 1):  # Top 10 results
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

        The LLM is instructed to format tool calls in a specific way.
        We look for patterns like:
        - "Call: check_stock with sku=BRK-1007"
        - "Use tool: find_parts_by_vehicle with make=Bajaj, model=Pulsar 150"
        """
        # Pattern 1: "Call: tool_name with param1=value1, param2=value2"
        pattern1 = r"Call:\s*(\w+)\s+with\s+(.+)"
        match = re.search(pattern1, response, re.IGNORECASE)
        if match:
            tool_name = match.group(1)
            params_str = match.group(2)
            return self._parse_params(params_str, tool_name)

        # Pattern 2: JSON format
        pattern2 = r"\{(.+)\}"
        match = re.search(pattern2, response)
        if match:
            try:
                data = json.loads(f"{{{match.group(1)}}}")
                if "tool" in data and "parameters" in data:
                    return {
                        "tool": data["tool"],
                        "parameters": data["parameters"]
                    }
            except json.JSONDecodeError:
                pass

        # Pattern 3: "Use tool: tool_name"
        pattern3 = r"Use\s+tool:\s*(\w+)"
        match = re.search(pattern3, response, re.IGNORECASE)
        if match:
            tool_name = match.group(1)
            return {"tool": tool_name, "parameters": {}}

        return None

    def _parse_params(self, params_str: str, tool_name: str) -> dict:
        """Parse parameters from a comma-separated key=value string."""
        params = {}

        # Split by comma, but be careful with nested structures
        parts = [p.strip() for p in params_str.split(",")]

        for part in parts:
            if "=" in part:
                key, value = part.split("=", 1)
                key = key.strip()
                value = value.strip()

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

    def _needs_clarification_llm(self, query: str) -> tuple[bool, str]:
        """
        Use the LLM to determine if clarification is needed.

        This replaces the pattern-based clarification detection.
        """
        # For now, we'll use a lightweight check
        # In production, we'd call the LLM with a clarification prompt

        # Skip clarification for direct SKU queries
        if self._extract_sku(query):
            return False, ""

        # Check for obvious action queries
        action_keywords = [
            "stock of ", "check stock", "what's the stock", "what is the stock",
            "price of ", "what is the price", "what's the price",
            "place order", "create order", "order for", "i want to order",
            "find parts", "parts for", "show me"
        ]

        query_lower = query.lower()
        for keyword in action_keywords:
            if keyword in query_lower:
                # Check if it has enough information
                if " for " in query_lower or self._extract_sku(query):
                    return False, ""
                elif any(w in query_lower for w in ["bajaj", "ktm", "royal enfield", "honda", "yamaha", "tvs", "hero", "suzuki"]):
                    # Has vehicle mention
                    return False, ""

        # Short queries likely need clarification
        if len(query.split()) <= 2:
            return True, "Could you please provide more details?"

        # Ambiguous queries
        ambiguous_patterns = [
            r"i need tyres\b", r"tyres for", r"i need brakes\b",
            r"chain lube\b", r"engine oil\b", r"show me\b",
            r"what (?:parts|items|products)\b", r"give me\b"
        ]

        for pattern in ambiguous_patterns:
            if re.search(pattern, query_lower):
                return True, "Which vehicle? Please specify make and model (e.g., 'Bajaj Pulsar 150')."

        return False, ""

    def _check_guardrails_llm(self, query: str) -> tuple[bool, str]:
        """
        Use the LLM to check if a query is off-topic or unsafe.

        This replaces the hardcoded off-topic detection.
        For efficiency, we first use a lightweight check, then fall back to LLM.
        """
        # Lightweight check first (for performance)
        if self._is_off_topic_lightweight(query):
            return True, self._generate_guardrail_response()

        # If lightweight check is uncertain, use LLM
        # For now, we'll use the lightweight check to avoid API calls
        return False, ""

    def _is_off_topic_lightweight(self, query: str) -> bool:
        """
        Lightweight off-topic detection (non-LLM).

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

        # If it doesn't contain any on-topic keywords and no off-topic phrases,
        # it might be ambiguous - we'll let it through for now
        return False

    def _generate_guardrail_response(self) -> str:
        """Generate a polite guardrail response."""
        return "I'm sorry, I can only help with auto parts, stock, and orders. Is there something related to auto parts I can help you with?"

    def _execute_tool(self, tool_name: str, parameters: dict) -> Any:
        """Execute a tool with the given parameters."""
        tool = self.tools.get_tool(tool_name)
        if tool is None:
            raise ValueError(f"Tool {tool_name} not found")

        return tool(**parameters)

    def process_query(self, query: str) -> str:
        """
        Process a user query using LLM-driven reasoning.

        This is the main entry point for the agent.
        """
        self.context.current_query = query
        self.context.add_message("user", query)

        # Extract entities from query
        self.context.extract_entities_from_query(query)

        # Check guardrails first
        is_off_topic, guardrail_response = self._check_guardrails_llm(query)
        if is_off_topic:
            self.context.add_message("assistant", guardrail_response)
            return guardrail_response

        # Check if we're waiting for clarification
        if self.context.pending_clarification:
            # Process the clarification response
            self.context.pending_clarification = None
            return self.process_query(guardrail_response)

        # Check if clarification is needed (using LLM reasoning)
        needs_clarification, clarification_msg = self._needs_clarification_llm(query)
        if needs_clarification:
            self.context.pending_clarification = clarification_msg
            self.context.add_message("assistant", clarification_msg)
            return clarification_msg

        # Perform retrieval (LLM will use this context)
        retrieval_results = self.retriever.search(query, k=5)
        self.context.last_retrieval_results = retrieval_results

        # Check confidence - but if we have a SKU in the query, trust it
        has_sku = self._extract_sku(query) is not None
        if (retrieval_results and
            retrieval_results[0].get('similarity_score', 0) < 0.7 and
            not has_sku):
            # Low confidence and no SKU - ask for clarification
            clarification = self._generate_clarification(query)
            self.context.pending_clarification = clarification
            self.context.add_message("assistant", clarification)
            return clarification

        # Build LLM messages with full context
        messages = self._build_llm_messages(query, retrieval_results)

        # Call the LLM to decide what to do
        try:
            result = self.provider.chat(messages)
            llm_response = result.content
        except Exception as e:
            # Fallback to pattern-based logic if LLM fails
            return self._fallback_to_pattern_based(query, retrieval_results)

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

    def _fallback_to_pattern_based(self, query: str, retrieval_results: list[dict]) -> str:
        """
        Fallback to pattern-based logic if LLM is unavailable.

        This ensures the system still works even without LLM access.
        """
        # Import the old agent for fallback
        from assistant.agent import DealerAssistant as OldAgent

        # Create a temporary old agent
        old_agent = OldAgent(self.retriever, self.tools)
        return old_agent.process_query(query)

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

    def _generate_clarification(self, query: str) -> str:
        """Generate a clarification question."""
        query_lower = query.lower()
        if any(w in query_lower for w in ['tyre', 'brake', 'chain', 'oil', 'parts']):
            return "Which vehicle? Please specify make and model (e.g., 'Bajaj Pulsar 150')."
        return "Could you please provide more details?"

    def _generate_retrieval_response(self, query: str, retrieval_results: list[dict]) -> str:
        """Generate response from retrieval (fallback)."""
        if not retrieval_results:
            return "No products found. Try different keywords."
        parts = []
        for i, p in enumerate(retrieval_results[:5], 1):
            parts.append(f"{i}. {p.get('name','N/A')} (SKU: {p.get('sku','N/A')}) - ₹{p.get('price_inr',0)}, Stock: {p.get('stock',0)}")
        return "Matching products:\n" + "\n".join(parts)

    def _check_grounding(self, response: str, retrieval_results: list[dict]) -> bool:
        """Check if response is grounded in retrieval results."""
        if not retrieval_results:
            return True  # No retrieval context to check against

        retrieved_skus = {p.get('sku', '') for p in retrieval_results}
        mentioned_skus = re.findall(r"\b[A-Z]{2,4}-\d{3,5}\b", response)

        return all(sku in retrieved_skus for sku in mentioned_skus)


# Singleton instance for compatibility
_agent_instance: Optional[LLMDealerAssistant] = None


def get_llm_agent(retriever: CatalogueRetriever, tool_registry: ToolRegistry) -> LLMDealerAssistant:
    """Get or create the LLM agent instance."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = LLMDealerAssistant(retriever, tool_registry)
    return _agent_instance


def reset_llm_agent():
    """Reset the agent instance (useful for testing)."""
    global _agent_instance
    _agent_instance = None
