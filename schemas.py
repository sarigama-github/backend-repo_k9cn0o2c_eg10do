"""
Database Schemas

Define your MongoDB collection schemas here using Pydantic models.
These schemas are used for data validation in your application.

Each Pydantic model represents a collection in your database.
Model name is converted to lowercase for the collection name:
- User -> "user" collection
- Product -> "product" collection
- BlogPost -> "blogs" collection
"""

from pydantic import BaseModel, Field
from typing import Optional, List

# Example schemas (replace with your own):

class User(BaseModel):
    """
    Users collection schema
    Collection name: "user" (lowercase of class name)
    """
    name: str = Field(..., description="Full name")
    email: str = Field(..., description="Email address")
    address: str = Field(..., description="Address")
    age: Optional[int] = Field(None, ge=0, le=120, description="Age in years")
    is_active: bool = Field(True, description="Whether user is active")

class Product(BaseModel):
    """
    Products collection schema
    Collection name: "product" (lowercase of class name)
    """
    title: str = Field(..., description="Product title")
    description: Optional[str] = Field(None, description="Product description")
    price: float = Field(..., ge=0, description="Price in dollars")
    category: str = Field(..., description="Product category")
    in_stock: bool = Field(True, description="Whether product is in stock")

# Sudoku collections

class Gamestate(BaseModel):
    """
    Active Sudoku game state per device
    Collection name: "gamestate"
    """
    device_id: str = Field(..., description="Anonymous device identifier")
    difficulty: str = Field(..., description="easy | medium | hard")
    puzzle: List[List[int]] = Field(..., description="Initial puzzle grid with 0 for empty")
    solution: List[List[int]] = Field(..., description="Solved grid")
    current: List[List[int]] = Field(..., description="Current grid values (0 for empty)")
    fixed: List[List[bool]] = Field(..., description="Fixed given cells")
    notes: List[List[List[int]]] = Field(..., description="Candidates per cell")
    mistakes: int = Field(0, ge=0)
    elapsed_seconds: int = Field(0, ge=0)
    is_completed: bool = Field(False)

class Stats(BaseModel):
    """
    Gameplay statistics per device
    Collection name: "stats"
    """
    device_id: str = Field(...)
    games_played: int = Field(0, ge=0)
    games_won: int = Field(0, ge=0)
    best_time_seconds: Optional[int] = Field(None, ge=0)
    last_difficulty: Optional[str] = Field(None)
    total_mistakes: int = Field(0, ge=0)
