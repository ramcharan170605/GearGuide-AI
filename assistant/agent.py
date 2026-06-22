"""
LLM-First Agent for GearGuide-AI Dealer Assistant
TRUE LLM-first architecture. NO pattern-based decision making.
"""

import re
import json
from typing import Optional, Any
from dataclasses import dataclass, field

from assistant.llm_provider import Message, get_llm_provider, LLMConfig
from assistant.retrieval import CatalogueRetriever
from assistant.tools import ToolRegistry, CheckStockResult, CreateOrderResult, OrderItem


@dataclass
class ConversationContext:
    history: list[Message] = field(default_factory=list)
    current_query: Optional[str] = None
    last_tool_result: Optional[dict] = None
    last_retrieval_results: Optional[list[dict]] = None
    mentioned_skus: list[str] = field(default_factory=list)
    conversation_metadata: dict = field(default_factory=dict)

    def add_message(self, role: str, content: str):
        self.history.append(Message(role=role, content=content))

    def extract_skus(self, query: str):
        skus = re.findall(r"\b[A-Z]{2,4}-\d{3,5}\b", query)
        self.mentioned_skus.extend(skus)

    def get_context_summary(self, max_tokens: int = 2000) -> str:
        if not self.history:
            return ""
        context_parts = []
        for msg in self.history[-10:]:
            context_parts.append(f"{msg.role.upper()}: {msg.content}")
        if self.mentioned_skus:
            context_parts.append(f"Mentioned SKUs: {', '.join(self.mentioned_skus)}")
        return "\n".join(context_parts[-20:])

    def clear(self):
        self.history = []
        self.current_query = None
        self.last_tool_result = None
        self.last_retrieval_results = None
        self.mentioned_skus = []
        self.conversation_metadata = {}


class DealerAssistant:
    def __init__(self, retriever: CatalogueRetriever, tool_registry: ToolRegistry, config: Optional[LLMConfig] = None):
        self.retriever = retriever
        self.tools = tool_registry
        self.config = config or LLMConfig()
        self.context = ConversationContext()
        self.confidence_threshold: float = 0.7

        self.provider = get_llm_provider(self.config)
        if not self.provider.is_available():
            raise RuntimeError(
                "LLM provider is required. Configure GEMINI_API_KEY or OPENAI_API_KEY in .env"
            )

        self._tool_descriptions = self._build_tool_descriptions()

    def _build_tool_descriptions(self) -> list[dict]:
        descriptions = []
        for name in self.tools.list_tools():
            info = self.tools.get_tool_info(name)
            descriptions.append({
                "name": name,
                "description": info.get("description", ""),
                "parameters": info.get("schema", {}).get("properties", {}),
                "required": info.get("schema", {}).get("required", [])
            })
        return descriptions

    def _format_retrieval_results(self, results: list[dict]) -> str:
        if not results:
            return "No retrieval results."
        lines = []
        for i, p in enumerate(results[:10], 1):
            lines.append(f"Result {i}: SKU={p.get('sku','N/A')}, Name={p.get('name','N/A')}, "
                        f"Price=₹{p.get('price_inr',0)}, Stock={p.get('stock',0)}, "
                        f"Category={p.get('category','N/A')}, Vehicle={p.get('vehicle_fitment','N/A')}")
        return "\n".join(lines)

    def _format_tool_descriptions(self) -> str:
        return json.dumps(self._tool_descriptions, indent=2)

    def _parse_llm_response(self, response: str) -> dict:
        response = response.strip()
        try:
            if response.startswith('{') and response.endswith('}'):
                return json.loads(response)
        except json.JSONDecodeError:
            pass
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass
        return {"action": "response", "answer": response}

    def _execute_tool(self, tool_name: str, parameters: dict) -> Any:
        tool = self.tools.get_tool(tool_name)
        if tool is None:
            raise ValueError(f"Tool {tool_name} not found")
        return tool(**parameters)

    def _format_tool_result(self, tool_name: str, result: Any) -> str:
        if tool_name == "check_stock":
            r = result
            status = "In Stock" if r.in_stock else "Out of Stock"
            return f"{r.name} (SKU: {r.sku}) - ₹{r.price_inr}, Stock: {r.stock}, Status: {status}"
        elif tool_name == "create_order":
            r = result
            items = "\n".join(f" - {i.sku}: {i.quantity}" for i in r.items)
            return f"Order {r.order_id} for {r.dealer} - Total: ₹{r.total_inr}\nItems:\n{items}"
        elif tool_name == "find_parts_by_vehicle":
            results = result
            if not results:
                return "No parts found."
            parts = []
            for i, p in enumerate(results[:10], 1):
                parts.append(f"{i}. {p.get('name','N/A')} (SKU: {p.get('sku','N/A')}) - ₹{p.get('price_inr',0)}, Stock: {p.get('stock',0)}")
            return "Matching products:\n" + "\n".join(parts)
        return str(result)

    def _check_grounding(self, response: str, retrieval_results: list[dict]) -> bool:
        if not retrieval_results:
            return True
        retrieved_skus = {p.get('sku', '') for p in retrieval_results}
        mentioned_skus = re.findall(r"\b[A-Z]{2,4}-\d{3,5}\b", response)
        return all(sku in retrieved_skus for sku in mentioned_skus)

    def process_query(self, query: str) -> str:
        self.context.current_query = query
        self.context.add_message("user", query)
        self.context.extract_skus(query)

        retrieval_results = self.retriever.search(query, k=5)
        self.context.last_retrieval_results = retrieval_results
        retrieval_context = self._format_retrieval_results(retrieval_results)
        conversation_context = self.context.get_context_summary(2000)

        llm_response = self._analyze_with_llm(query, retrieval_context, conversation_context)
        parsed = self._parse_llm_response(llm_response)
        action = parsed.get("action")

        if action == "guardrail":
            response = parsed.get("message", "I'm sorry, I can only help with auto parts, stock, and orders.")
            self.context.add_message("assistant", response)
            return response

        elif action == "clarification":
            response = parsed.get("question", "Could you clarify?")
            self.context.pending_clarification = response
            self.context.add_message("assistant", response)
            return response

        elif action == "tool":
            tool_name = parsed.get("tool")
            parameters = parsed.get("parameters", {})
            if tool_name and parameters:
                try:
                    tool_result = self._execute_tool(tool_name, parameters)
                    self.context.last_tool_result = {"tool": tool_name, "result": tool_result}
                    formatted_result = self._format_tool_result(tool_name, tool_result)
                    # Feed back to the LLM to generate a natural, grounded response
                    response = self._generate_response_with_tool_result(
                        query, tool_name, formatted_result, retrieval_context, conversation_context
                    )
                    self.context.add_message("assistant", response)
                    return response
                except Exception as e:
                    return f"Error: {str(e)}"
            response = parsed.get("answer", "I didn't understand.")
            self.context.add_message("assistant", response)
            return response

        else:
            response = parsed.get("answer", "I didn't understand.")
            if retrieval_results and not self._check_grounding(response, retrieval_results):
                grounded = self._generate_grounded_response(query, retrieval_context, conversation_context)
                parsed_grounded = self._parse_llm_response(grounded)
                response = parsed_grounded.get("answer", response)
            self.context.add_message("assistant", response)
            return response

    def _analyze_with_llm(self, query: str, retrieval_context: str, conversation_context: str) -> str:
        system_prompt = f"""You are GearGuide-AI, an auto parts assistant. Analyze the query and respond with JSON.

Available tools: {self._format_tool_descriptions()}

Retrieval Context:
{retrieval_context}

Conversation:
{conversation_context}

Respond with ONLY JSON in one format:
- Tool call: {{"action": "tool", "tool": "check_stock", "parameters": {{"sku": "BRK-1007"}}}}
- Clarification: {{"action": "clarification", "question": "Which vehicle?"}}
- Direct answer: {{"action": "response", "answer": "Your answer grounded in context"}}
- Guardrail: {{"action": "guardrail", "message": "I'm sorry..."}}

Rules:
1. If off-topic (not auto parts/stock/orders): return guardrail
2. If query is unclear: return clarification
3. If tool would help: return tool call with correct parameters
4. Otherwise: return direct answer grounded in retrieval context
5. ALL product mentions MUST come from retrieval context
"""
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=f"Query: {query}")
        ]
        result = self.provider.chat(messages)
        return result.content

    def _generate_grounded_response(self, query: str, retrieval_context: str, conversation_context: str) -> str:
        system_prompt = f"""Generate a response grounded ONLY in the retrieval context below.

Retrieval Context:
{retrieval_context}

Conversation:
{conversation_context}

Respond with JSON: {{"action": "response", "answer": "..."}}

Rules:
- ONLY mention products from the retrieval context
- If you cannot ground the response, ask for clarification
"""
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=f"Query: {query}")
        ]
        result = self.provider.chat(messages)
        return result.content

    def _generate_response_with_tool_result(self, query: str, tool_name: str, tool_result: str, retrieval_context: str, conversation_context: str) -> str:
        system_prompt = f"""You are GearGuide-AI, an auto parts assistant. Respond with JSON.

Retrieval Context:
{retrieval_context}

Conversation History:
{conversation_context}

Tool Executed: {tool_name}
Tool Result:
{tool_result}

Respond with ONLY JSON in this format:
{{"action": "response", "answer": "Your natural language response to the user query based on the tool result"}}
"""
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=f"Query: {query}")
        ]
        result = self.provider.chat(messages)
        parsed = self._parse_llm_response(result.content)
        return parsed.get("answer", result.content)
