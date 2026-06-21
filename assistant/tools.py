"""
Tool implementations for Dealer Assistant

Implements required tools:
- check_stock: Look up availability for a product (TOOL-001)
- find_parts_by_vehicle: Find parts by make/model/year (TOOL-003)
- create_order: Place order with structured output (TOOL-002)

Requirements: TOOL-001 to TOOL-007
Tasks: P2-T002, P2-T003, P2-T004
"""

from typing import Optional
from pydantic import BaseModel, Field, validator


class CheckStockResult(BaseModel):
    """Result of check_stock tool."""
    sku: str
    name: str
    stock: int
    in_stock: bool
    price_inr: int


class OrderItem(BaseModel):
    """Line item for an order."""
    sku: str
    quantity: int = Field(..., gt=0)


class CreateOrderResult(BaseModel):
    """Result of create_order tool (TOOL-005, TOOL-007)."""
    order_id: str
    dealer: str
    items: list[OrderItem]
    total_inr: int
    status: str = "pending"


class ToolRegistry:
    """
    Registry for all available tools.
    
    Allows the agent to discover and call tools by name.
    Requirements: TOOL-004, TOOL-006
    Task: P2-T001
    """
    
    def __init__(self):
        self.tools: dict[str, callable] = {}
        self.schemas: dict[str, dict] = {}
    
    def register(self, name: str, func: callable, schema: dict = None):
        """Register a tool with the registry."""
        self.tools[name] = func
        if schema:
            self.schemas[name] = schema
    
    def get_tool(self, name: str) -> Optional[callable]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def list_tools(self) -> list[str]:
        """List all available tool names."""
        return list(self.tools.keys())


def check_stock(sku: str, catalogue_data: dict) -> CheckStockResult:
    """
    Look up stock availability for a product.
    
    Args:
        sku: Product SKU
        catalogue_data: Loaded catalogue data
    
    Returns:
        CheckStockResult: Stock information
    
    Requirements: TOOL-001, TOOL-005
    Task: P2-T002
    """
    raise NotImplementedError("Implement in P2-T002")


def find_parts_by_vehicle(make: str, model: str, year: Optional[str] = None, 
                          catalogue_data: dict, limit: int = 10) -> list[dict]:
    """
    Find parts that fit a specific vehicle.
    
    Args:
        make: Vehicle make
        model: Vehicle model
        year: Optional vehicle year
        catalogue_data: Loaded catalogue data
        limit: Maximum number of results
    
    Returns:
        list[dict]: Matching products
    
    Requirements: TOOL-003, TOOL-005
    Task: P2-T003
    """
    raise NotImplementedError("Implement in P2-T003")


def create_order(dealer: str, items: list[OrderItem], catalogue_data: dict) -> CreateOrderResult:
    """
    Create an order with structured output.
    
    Args:
        dealer: Dealer name
        items: List of order items
        catalogue_data: Loaded catalogue data
    
    Returns:
        CreateOrderResult: Order confirmation with structured output
    
    Requirements: TOOL-002, TOOL-005, TOOL-007
    Task: P2-T004
    """
    raise NotImplementedError("Implement in P2-T004")
