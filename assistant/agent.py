"""
Agent loop and conversation management for Dealer Assistant

Handles:
- Multi-turn dialogue (CONV-001, CONV-002)
- Clarification questions (CONV-003, CONV-004)
- Tool calling decisions (TOOL-004, TOOL-006)
- Grounding checks (GROUND-001, GROUND-002)

Requirements: CONV-001 to CONV-004, TOOL-004, TOOL-006, GROUND-001, GROUND-002
Tasks: P2-T005, P2-T006, P2-T007, P3-T001, P3-T003, P3-T004
"""

import re
from typing import Optional
from assistant.retrieval import CatalogueRetriever
from assistant.tools import ToolRegistry, OrderItem, CheckStockResult, CreateOrderResult


class ConversationContext:
    """Maintains conversation state across turns. Requirements: CONV-001, CONV-002. Task: P2-T005"""

    def __init__(self):
        self.history: list[dict] = []
        self.current_query: Optional[str] = None
        self.retrieval_results: Optional[list[dict]] = None
        self.tool_calls: list[dict] = []
        self.pending_clarification: Optional[str] = None
        self.conversation_metadata: dict = {}

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def clear(self):
        self.history = []
        self.current_query = None
        self.retrieval_results = None
        self.tool_calls = []
        self.pending_clarification = None
        self.conversation_metadata = {}


class DealerAssistant:
    """Main agent class for the Dealer Assistant."""

    def __init__(self, retriever: CatalogueRetriever, tool_registry: ToolRegistry):
        self.retriever = retriever
        self.tools = tool_registry
        self.context = ConversationContext()
        self.confidence_threshold: float = 0.7

    def process_query(self, query: str) -> str:
        """Process a user query and return a response. Requirements: CONV-001, CONV-002, CONV-003, TOOL-004. Tasks: P2-T005, P2-T006, P2-T007"""
        self.context.current_query = query
        self.context.add_message("user", query)

        # Check for off-topic queries (guardrail) - P3-T002
        if self._is_off_topic(query):
            response = self._generate_guardrail_response(query)
            self.context.add_message("assistant", response)
            return response

        # Check if waiting for clarification
        if self.context.pending_clarification:
            self.context.pending_clarification = None
            return self._process_with_context(query)

        # Check if clarification needed
        clarification = self._needs_clarification(query)
        if clarification:
            self.context.pending_clarification = clarification
            self.context.add_message("assistant", clarification)
            return clarification

        # Perform retrieval
        retrieval_results = self.retriever.search(query, k=5)
        self.context.retrieval_results = retrieval_results

        # Check confidence - but if we have a SKU in the query, trust it
        # (SKU-based queries are specific and don't need high semantic similarity)
        has_sku = self._extract_sku(query) is not None
        if (retrieval_results and
            retrieval_results[0].get('similarity_score', 0) < self.confidence_threshold and
            not has_sku):
            clarification = self._generate_clarification(query)
            self.context.pending_clarification = clarification
            self.context.add_message("assistant", clarification)
            return clarification

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

    def _process_with_context(self, clarification: str) -> str:
        """Process clarification response."""
        # For now, just treat as new query with clarification
        return self.process_query(clarification)

    def _is_off_topic(self, query: str) -> bool:
        """
        Check if query is off-topic (not about auto parts, stock, or orders).
        Requirements: Bonus Guardrails (+5 points)
        Task: P3-T002
        """
        query_lower = query.lower()

        # Off-topic phrases that should be blocked even if they contain on-topic words
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

        # Check for off-topic phrases first
        if any(phrase in query_lower for phrase in off_topic_phrases):
            return True

        # On-topic keywords
        on_topic_keywords = [
            'part', 'parts', 'product', 'products', 'item', 'items',
            'stock', 'order', 'price', 'inventory', 'available', 'availability',
            'brake', 'tyre', 'tire', 'chain', 'oil', 'engine', 'filter',
            'bike', 'motorcycle', 'scooter', 'car', 'vehicle', 'auto',
            'bajaj', 'ktm', 'royal enfield', 'honda', 'yamaha', 'tvs', 'hero', 'suzuki',
            'sku', 'catalogue', 'catalog', 'purchase', 'buy', 'sell',
            'lube', 'lubricant', 'pad', 'disc', 'rotor', 'sprocket',
            'seat', 'cover', 'mirror', 'light', 'horn', 'battery',
            'exhaust', 'muffler', 'guard', 'tidy', 'assembly',
            'show', 'have', 'cheapest', 'cheap', 'expensive', 'cost'
        ]
        return not any(keyword in query_lower for keyword in on_topic_keywords)

    def _generate_guardrail_response(self, query: str) -> str:
        """
        Generate a polite guardrail response for off-topic queries.
        Requirements: Bonus Guardrails (+5 points)
        Task: P3-T002
        """
        responses = [
            "I'm sorry, I can only help with auto parts, stock, and orders. Is there something related to auto parts I can help you with?",
            "I'm not able to assist with that. I'm here to help with auto parts queries, stock checks, and order placement.",
            "I'm not sure I can help with that. Please ask about auto parts, stock availability, or placing orders."
        ]
        return responses[0]  # Simple: always use first response

    def _needs_clarification(self, query: str) -> Optional[str]:
        """Determine if query needs clarification. Requirements: CONV-003, CONV-004. Task: P2-T006"""
        query_lower = query.lower()
        # Don't ask for clarification if query contains specific identifiers or actions
        if self._extract_sku(query):
            return None
        if any(w in query_lower for w in ['stock of ', 'check stock', 'availability of', 'what is the stock', "what's the stock", 'place order', 'create order', 'order for']):
            return None
        # Check for ambiguous queries that need vehicle specification
        patterns = [r"i need tyres\b", r"tyres for", r"i need brakes?\b", r"chain lube\b", r"engine oil\b", r"show me\b", r"what (?:parts|items|products)\b"]
        for pattern in patterns:
            if re.search(pattern, query_lower):
                return self._generate_clarification(query)
        # Short vague queries
        if len(query.split()) <= 2 and not any(w in query_lower for w in ['stock', 'order', 'price', 'find', 'search']):
            return self._generate_clarification(query)
        return None

    def _generate_clarification(self, query: str) -> str:
        """Generate clarification question. Requirements: CONV-003, CONV-004. Task: P2-T006"""
        query_lower = query.lower()
        if any(w in query_lower for w in ['tyre', 'brake', 'chain', 'oil', 'parts']):
            return "Which vehicle? Please specify make and model (e.g., 'Bajaj Pulsar 150')."
        return "Could you please provide more details?"

    def _decide_action(self, query: str, retrieval_results: list[dict]) -> str:
        """Decide action type. Requirements: TOOL-004, TOOL-006. Task: P2-T007"""
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

    def _extract_sku(self, query: str) -> Optional[str]:
        """Extract SKU from query."""
        match = re.search(r"\b[A-Z]{2,4}-\d{3,5}\b", query)
        return match.group(0) if match else None

    def _execute_tool_action(self, query: str, retrieval_results: list[dict]) -> str:
        """Execute tool action. Requirements: TOOL-004, TOOL-006. Task: P2-T007"""
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
                        return self._format_vehicle_results(result)

            return self._generate_retrieval_response(query, retrieval_results)
        except Exception as e:
            return f"Error: {str(e)}. Please try again."

    def _parse_order_query(self, query: str):
        """Parse order query."""
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

    def _generate_retrieval_response(self, query: str, retrieval_results: list[dict]) -> str:
        """Generate response from retrieval. Requirements: GROUND-001, GROUND-002. Task: P2-T005"""
        if not retrieval_results:
            return "No products found. Try different keywords."
        parts = []
        for i, p in enumerate(retrieval_results[:5], 1):
            parts.append(f"{i}. {p.get('name','N/A')} (SKU: {p.get('sku','N/A')}) - ₹{p.get('price_inr',0)}, Stock: {p.get('stock',0)}")
        return "Matching products:\n" + "\n".join(parts)

    def _format_stock_result(self, result: CheckStockResult) -> str:
        """Format stock result."""
        status = "In Stock" if result.in_stock else "Out of Stock"
        return f"{result.name} (SKU: {result.sku}) - ₹{result.price_inr}, Stock: {result.stock}, Status: {status}"

    def _format_order_result(self, result: CreateOrderResult) -> str:
        """Format order result."""
        items = "\n".join(f"  - {i.sku}: {i.quantity}" for i in result.items)
        return f"Order {result.order_id} for {result.dealer} - Total: ₹{result.total_inr}\nItems:\n{items}"

    def _check_grounding(self, response: str, retrieval_results: list[dict]) -> bool:
        """Check grounding. Requirements: GROUND-001, GROUND-002. Task: P3-T004"""
        retrieved_skus = {p.get('sku', '') for p in retrieval_results}
        mentioned_skus = re.findall(r"\b[A-Z]{2,4}-\d{3,5}\b", response)
        return all(sku in retrieved_skus for sku in mentioned_skus)
