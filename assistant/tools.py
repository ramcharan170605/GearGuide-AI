"""
Tool implementations for Dealer Assistant

Implements required tools:
- check_stock: Look up availability for a product (TOOL-001)
- find_parts_by_vehicle: Find parts by make/model/year (TOOL-003)
- create_order: Place order with structured output (TOOL-002)

The tools are now designed to be called by the LLM agent with proper
parameter validation and structured output.

Requirements: TOOL-001 to TOOL-007
Tasks: P2-T001, P2-T002, P2-T003, P2-T004
"""

import uuid
import numpy as np
from typing import Optional, Any, Callable
from pydantic import BaseModel, Field, field_validator
import pandas as pd


# ============================================================================
# Pydantic Models for Structured Output (TOOL-005, TOOL-007)
# ============================================================================

class CheckStockResult(BaseModel):
    """
    Result of check_stock tool.
    Structured output as required by TOOL-005 and TOOL-007.
    """
    sku: str = Field(..., description="Product SKU")
    name: str = Field(..., description="Product name")
    stock: int = Field(..., description="Current stock quantity")
    in_stock: bool = Field(..., description="Whether product is in stock")
    price_inr: int = Field(..., description="Price in INR")
    category: str = Field(..., description="Product category")


class OrderItem(BaseModel):
    """Line item for an order."""
    sku: str = Field(..., description="Product SKU")
    quantity: int = Field(..., gt=0, description="Quantity (must be positive)")

    @field_validator('quantity')
    def quantity_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Quantity must be positive')
        return v


class CreateOrderResult(BaseModel):
    """
    Result of create_order tool.
    Structured output as required by TOOL-005 and TOOL-007.
    """
    order_id: str = Field(..., description="Unique order identifier")
    dealer: str = Field(..., description="Dealer name")
    items: list[OrderItem] = Field(..., description="List of order items")
    total_inr: int = Field(..., description="Total amount in INR")
    status: str = Field(default="pending", description="Order status")


class FindPartsResult(BaseModel):
    """
    Result of find_parts_by_vehicle tool.
    Structured output for vehicle part search.
    """
    make: str = Field(..., description="Vehicle make")
    model: str = Field(..., description="Vehicle model")
    year: Optional[str] = Field(default=None, description="Vehicle year (optional)")
    parts: list[dict] = Field(..., description="List of matching parts")


# ============================================================================
# Tool Registry (P2-T001: TOOL-004, TOOL-006)
# ============================================================================

class ToolRegistry:
    """
    Registry for all available tools.

    Allows the agent to discover and call tools by name.
    Now supports both direct function registration and factory functions
    that receive catalogue data.

    Requirements: TOOL-004, TOOL-006
    Task: P2-T001
    """

    def __init__(self):
        self.tools: dict[str, callable] = {}
        self.schemas: dict[str, dict] = {}
        self.descriptions: dict[str, str] = {}
        self._catalogue_data: Optional[pd.DataFrame] = None

    def set_catalogue(self, catalogue_data: pd.DataFrame):
        """Set the catalogue data for tools that need it."""
        self._catalogue_data = catalogue_data

    def register(self, name: str, func: callable, schema: dict = None,
                 description: str = "", requires_catalogue: bool = False):
        """
        Register a tool with the registry.

        Args:
            name: Tool name
            func: Tool function
            schema: JSON schema for tool parameters
            description: Tool description
            requires_catalogue: Whether the tool needs catalogue data
        """
        # If the tool requires catalogue, wrap it
        if requires_catalogue and self._catalogue_data is not None:
            def wrapped_func(*args, **kwargs):
                return func(*args, catalogue_data=self._catalogue_data, **kwargs)
            self.tools[name] = wrapped_func
        else:
            self.tools[name] = func

        if schema:
            self.schemas[name] = schema
        if description:
            self.descriptions[name] = description

    def get_tool(self, name: str) -> Optional[callable]:
        """
        Get a tool by name.

        Args:
            name: Tool name

        Returns:
            callable: Tool function or None if not found
        """
        return self.tools.get(name)

    def list_tools(self) -> list[str]:
        """List all available tool names."""
        return list(self.tools.keys())

    def get_tool_info(self, name: str) -> dict[str, Any]:
        """
        Get information about a tool.

        Args:
            name: Tool name

        Returns:
            dict: Tool information
        """
        return {
            "name": name,
            "description": self.descriptions.get(name, ""),
            "schema": self.schemas.get(name, {})
        }


# ============================================================================
# Tool Implementations
# ============================================================================

def check_stock(sku: str, catalogue_data: Optional[pd.DataFrame] = None) -> CheckStockResult:
    """
    Look up stock availability for a product.

    Args:
        sku: Product SKU
        catalogue_data: Loaded catalogue DataFrame

    Returns:
        CheckStockResult: Stock information as structured output

    Requirements: TOOL-001, TOOL-005, TOOL-007
    Task: P2-T002
    """
    if catalogue_data is None:
        raise ValueError("catalogue_data is required for check_stock")

    # Find product by SKU
    product = catalogue_data[catalogue_data['sku'] == sku]

    if product.empty:
        raise ValueError(f"Product with SKU {sku} not found in catalogue")

    row = product.iloc[0]

    return CheckStockResult(
        sku=sku,
        name=row['name'],
        stock=int(row['stock']),
        in_stock=row['stock'] > 0,
        price_inr=int(row['price_inr']),
        category=row['category']
    )


def find_parts_by_vehicle(make: str, model: str, catalogue_data: Optional[pd.DataFrame] = None,
                          year: Optional[str] = None, limit: int = 10) -> list[dict]:
    """
    Find parts that fit a specific vehicle.

    Args:
        make: Vehicle make (e.g., "Bajaj")
        model: Vehicle model (e.g., "Pulsar 150")
        catalogue_data: Loaded catalogue DataFrame
        year: Optional vehicle year
        limit: Maximum number of results

    Returns:
        list[dict]: Matching products as dictionaries

    Requirements: TOOL-003, TOOL-005
    Task: P2-T003
    """
    if catalogue_data is None:
        raise ValueError("catalogue_data is required for find_parts_by_vehicle")

    # Build vehicle string
    vehicle_str = f"{make} {model}"
    if year:
        vehicle_str += f" {year}"

    # Filter by exact match on vehicle_fitment
    exact_matches = catalogue_data[
        catalogue_data['vehicle_fitment'].str.lower() == vehicle_str.lower()
    ]

    if not exact_matches.empty:
        # Return exact matches
        results = exact_matches.head(limit).to_dict('records')
        # Convert numpy types to Python types
        for r in results:
            for key, value in r.items():
                if pd.isna(value):
                    r[key] = None
                elif isinstance(value, (np.integer, np.floating)):
                    r[key] = int(value) if isinstance(value, np.integer) else float(value)
        return results

    # If no exact matches, try partial match (make and model in any order)
    partial_matches = catalogue_data[
        catalogue_data['vehicle_fitment'].str.lower().str.contains(make.lower()) &
        catalogue_data['vehicle_fitment'].str.lower().str.contains(model.lower())
    ]

    results = partial_matches.head(limit).to_dict('records')
    for r in results:
        for key, value in r.items():
            if pd.isna(value):
                r[key] = None
            elif isinstance(value, (np.integer, np.floating)):
                r[key] = int(value) if isinstance(value, np.integer) else float(value)

    return results


def create_order(dealer: str, items: list[OrderItem], catalogue_data: Optional[pd.DataFrame] = None) -> CreateOrderResult:
    """
    Create an order with structured output.

    Args:
        dealer: Dealer name
        items: List of OrderItem objects
        catalogue_data: Loaded catalogue DataFrame

    Returns:
        CreateOrderResult: Order confirmation with structured output

    Requirements: TOOL-002, TOOL-005, TOOL-007
    Task: P2-T004
    """
    if catalogue_data is None:
        raise ValueError("catalogue_data is required for create_order")

    # Validate all SKUs exist and get prices
    total_inr = 0
    validated_items = []

    catalogue_skus = set(catalogue_data['sku'].tolist())

    for item in items:
        # Validate SKU exists
        if item.sku not in catalogue_skus:
            raise ValueError(f"Product with SKU {item.sku} not found in catalogue")

        # Get product info
        product = catalogue_data[catalogue_data['sku'] == item.sku].iloc[0]

        # Calculate subtotal
        subtotal = int(product['price_inr']) * item.quantity
        total_inr += subtotal

        validated_items.append(item)

    # Generate order ID
    order_id = f"ORD-{uuid.uuid4().hex[:8].upper()}"

    return CreateOrderResult(
        order_id=order_id,
        dealer=dealer,
        items=validated_items,
        total_inr=total_inr,
        status="pending"
    )


# ============================================================================
# Tool Factory Functions (for LLM agent)
# ============================================================================

def create_check_stock_tool(catalogue_data: pd.DataFrame) -> Callable[[str], CheckStockResult]:
    """Create a check_stock function with catalogue data bound."""
    return lambda sku: check_stock(sku, catalogue_data)


def create_find_parts_tool(catalogue_data: pd.DataFrame) -> Callable:
    """Create a find_parts_by_vehicle function with catalogue data bound."""
    return lambda make, model, year=None, limit=10: find_parts_by_vehicle(
        make, model, catalogue_data, year, limit
    )


def create_create_order_tool(catalogue_data: pd.DataFrame) -> Callable:
    """Create a create_order function with catalogue data bound."""
    return lambda dealer, items: create_order(dealer, items, catalogue_data)


# ============================================================================
# Initialize Default Tool Registry
# ============================================================================

def create_default_tool_registry(catalogue_data: pd.DataFrame) -> ToolRegistry:
    """
    Create a tool registry with all default tools registered.

    Args:
        catalogue_data: Loaded catalogue DataFrame

    Returns:
        ToolRegistry: Registry with all tools

    Task: P2-T001
    """
    registry = ToolRegistry()
    registry.set_catalogue(catalogue_data)

    # Register check_stock
    registry.register(
        name="check_stock",
        func=lambda sku: check_stock(sku, catalogue_data),
        description="Check stock availability for a product by SKU",
        schema={
            "type": "object",
            "properties": {
                "sku": {"type": "string", "description": "Product SKU"}
            },
            "required": ["sku"]
        }
    )

    # Register find_parts_by_vehicle
    registry.register(
        name="find_parts_by_vehicle",
        func=lambda make, model, year=None, limit=10: find_parts_by_vehicle(
            make, model, catalogue_data, year, limit
        ),
        description="Find parts that fit a specific vehicle by make, model, and optional year",
        schema={
            "type": "object",
            "properties": {
                "make": {"type": "string", "description": "Vehicle make"},
                "model": {"type": "string", "description": "Vehicle model"},
                "year": {"type": "string", "description": "Vehicle year (optional)"}
            },
            "required": ["make", "model"]
        }
    )

    # Register create_order
    registry.register(
        name="create_order",
        func=lambda dealer, items: create_order(dealer, items, catalogue_data),
        description="Create an order for a dealer with line items",
        schema={
            "type": "object",
            "properties": {
                "dealer": {"type": "string", "description": "Dealer name"},
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "sku": {"type": "string", "description": "Product SKU"},
                            "quantity": {"type": "integer", "minimum": 1, "description": "Quantity"}
                        },
                        "required": ["sku", "quantity"]
                    }
                }
            },
            "required": ["dealer", "items"]
        }
    )

    return registry


# ============================================================================
# Tests
# ============================================================================

if __name__ == "__main__":
    import pandas as pd

    print("Testing Tool Implementations...")

    # Load catalogue
    catalogue = pd.read_csv("catalogue.csv")
    print(f"✓ Loaded {len(catalogue)} products")

    # Test check_stock (P2-T002)
    stock_result = check_stock("BRK-1007", catalogue)
    print(f"✓ P2-T002: check_stock working")
    print(f" SKU: {stock_result.sku}, Stock: {stock_result.stock}, In Stock: {stock_result.in_stock}")

    # Test find_parts_by_vehicle (P2-T003)
    vehicle_results = find_parts_by_vehicle("Royal Enfield", "Meteor 350", catalogue_data=catalogue)
    print(f"✓ P2-T003: find_parts_by_vehicle working")
    print(f" Found {len(vehicle_results)} parts for Royal Enfield Meteor 350")

    # Test create_order (P2-T004)
    order_items = [OrderItem(sku="BRK-1007", quantity=5)]
    order_result = create_order("ABC Motors", order_items, catalogue)
    print(f"✓ P2-T004: create_order working")
    print(f" Order ID: {order_result.order_id}, Total: ₹{order_result.total_inr}")

    # Test tool registry (P2-T001)
    registry = create_default_tool_registry(catalogue)
    print(f"✓ P2-T001: Tool registry created with {len(registry.list_tools())} tools")
    print(f" Available tools: {registry.list_tools()}")

    print("\n✓ All Phase 2 tool tests passed!")
