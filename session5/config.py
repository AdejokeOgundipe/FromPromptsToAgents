"""
Configuration and Constants for Robotic Chef Platform
======================================================
Session 5: The Challenge - Configuration Module

This module centralizes all configuration settings, constants, and dietary
information for the Robotic Chef Platform.
"""

from typing import Dict, List, Set
from enum import Enum

# ============================================================================
# Environment & Service Configuration
# ============================================================================

# LLM Service Configuration
LLM_SERVICE_URL: str = "http://localhost:8000"
LLM_API_TOKEN: str = ""  # Will be loaded from .env
LLM_TIMEOUT_SECONDS: int = 60
LLM_MAX_RETRIES: int = 3

# MCP Server Configuration
MCP_TIMEOUT_SECONDS: int = 30
MCP_RETRY_ATTEMPTS: int = 2

# ============================================================================
# Budget & Nutrition Constraints
# ============================================================================

class DietaryFilter(str, Enum):
    """Supported dietary filters."""
    ANY = "any"
    VEGAN = "vegan"
    VEGETARIAN = "vegetarian"
    GLUTEN_FREE = "gluten_free"
    DAIRY_FREE = "dairy_free"
    NUT_FREE = "nut_free"


class MacroFocus(str, Enum):
    """Macronutrient focus areas."""
    PROTEIN = "protein"
    CARBOHYDRATE = "carbohydrate"
    BALANCED = "balanced"
    LOW_CALORIE = "low_calorie"
    HIGH_ENERGY = "high_energy"


# Budget Constraints (in GBP)
DEFAULT_BUDGET_GBP: float = 15.0
MIN_BUDGET_GBP: float = 5.0
MAX_BUDGET_GBP: float = 50.0
BUDGET_STEP_GBP: float = 1.0

# Serving Configuration
DEFAULT_SERVINGS: int = 2
MIN_SERVINGS: int = 1
MAX_SERVINGS: int = 10
DEFAULT_SERVINGS_MULTIPLIER: int = 2  # Base recipes are for this many servings

# ============================================================================
# Nutritional Targets
# ============================================================================

NUTRITIONAL_TARGETS: Dict[MacroFocus, Dict[str, float]] = {
    MacroFocus.PROTEIN: {
        "protein_g_per_serving": 25,
        "protein_pct_daily": 0.5,  # 50% of daily value
        "carbs_g_per_serving": 50,
        "calories_kcal_per_serving": 500,
    },
    MacroFocus.CARBOHYDRATE: {
        "protein_g_per_serving": 15,
        "carbs_g_per_serving": 80,
        "carbs_pct_daily": 0.6,  # 60% of daily value
        "calories_kcal_per_serving": 600,
    },
    MacroFocus.BALANCED: {
        "protein_g_per_serving": 20,
        "carbs_g_per_serving": 60,
        "fat_g_per_serving": 20,
        "calories_kcal_per_serving": 550,
    },
    MacroFocus.LOW_CALORIE: {
        "protein_g_per_serving": 25,
        "calories_kcal_per_serving": 350,
        "carbs_g_per_serving": 40,
    },
    MacroFocus.HIGH_ENERGY: {
        "calories_kcal_per_serving": 800,
        "carbs_g_per_serving": 100,
        "protein_g_per_serving": 20,
    },
}

# Daily Nutritional Values (reference)
DAILY_VALUES: Dict[str, float] = {
    "calories_kcal": 2000,
    "protein_g": 50,
    "carbs_g": 300,
    "fat_g": 70,
    "fiber_g": 25,
    "vitamin_c_mg": 90,
    "vitamin_a_iu": 2333,
    "iron_mg": 18,
    "calcium_mg": 1000,
    "potassium_mg": 3500,
}

# ============================================================================
# Allergens & Dietary Restrictions
# ============================================================================

ALLERGENS: Set[str] = {
    "peanuts",
    "tree_nuts",
    "milk",
    "eggs",
    "fish",
    "shellfish",
    "crustaceans",
    "soy",
    "wheat",
    "gluten",
    "sesame",
    "mustard",
}

DIETARY_RESTRICTIONS: Dict[DietaryFilter, Set[str]] = {
    DietaryFilter.VEGAN: {
        "contains_meat",
        "contains_fish",
        "contains_dairy",
        "contains_egg",
        "contains_honey",
    },
    DietaryFilter.VEGETARIAN: {
        "contains_meat",
        "contains_fish",
    },
    DietaryFilter.GLUTEN_FREE: {
        "contains_gluten",
        "contains_wheat",
    },
    DietaryFilter.DAIRY_FREE: {
        "contains_dairy",
    },
    DietaryFilter.NUT_FREE: {
        "contains_nuts",
        "contains_peanuts",
    },
}

# ============================================================================
# Meal Planning Algorithm
# ============================================================================

# Scoring weights for fit_budget() function
FITNESS_SCORE_WEIGHTS: Dict[str, float] = {
    "price_alignment": 0.25,  # How well it fits the budget
    "nutrition_match": 0.35,  # How well it matches nutritional targets
    "feasibility": 0.20,      # Ease of preparation
    "dietary_match": 0.20,    # Dietary preference compatibility
}

# Price variance tolerance (e.g., ±20% of budget)
PRICE_TOLERANCE_PCT: float = 0.20

# Scoring thresholds
POOR_SCORE_THRESHOLD: float = 0.4
FAIR_SCORE_THRESHOLD: float = 0.6
GOOD_SCORE_THRESHOLD: float = 0.8

# ============================================================================
# UI Configuration
# ============================================================================

# Streamlit UI settings
STREAMLIT_PAGE_WIDTH: str = "wide"
STREAMLIT_SIDEBAR_STATE: str = "expanded"

# Number of meal suggestions to display
TOP_MEAL_SUGGESTIONS: int = 3
TOP_DISHES_FOR_BUDGET: int = 3

# ============================================================================
# Shopping List Configuration
# ============================================================================

# Units for ingredient quantities
VOLUME_UNITS: List[str] = ["ml", "l", "cup", "tbsp", "tsp", "floz"]
WEIGHT_UNITS: List[str] = ["g", "kg", "oz", "lb"]
COUNT_UNITS: List[str] = ["piece", "pieces", "whole", "cloves", "slices"]

# Default quantities for common ingredients
DEFAULT_INGREDIENT_QUANTITIES: Dict[str, str] = {
    "salt": "to taste",
    "pepper": "to taste",
    "oil": "as needed",
    "water": "as needed",
    "butter": "as needed",
}

# ============================================================================
# Temperature & Precision Settings
# ============================================================================

# Temperature units
TEMPERATURE_UNIT: str = "Celsius"

# Cooking precision levels
PRECISION_LEVELS: Dict[str, int] = {
    "low": 1,           # ±20°C tolerance
    "medium": 2,        # ±10°C tolerance
    "high": 3,          # ±5°C tolerance
    "critical": 4,      # ±2°C tolerance
}

# ============================================================================
# Output & Display Configuration
# ============================================================================

# Logging configuration
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Maximum output lengths
MAX_AGENT_OUTPUT_LENGTH: int = 5000
MAX_DISH_NAME_LENGTH: int = 100
MAX_INGREDIENT_COUNT: int = 20

# ============================================================================
# Error Messages & Headers
# ============================================================================

ERROR_MESSAGES: Dict[str, str] = {
    "service_unavailable": "LLM service is currently unavailable. Please try again later.",
    "invalid_budget": "Budget must be between £{} and £{}.",
    "invalid_servings": "Servings must be between {} and {}.",
    "dish_not_found": "Dish '{}' not found in recipe database.",
    "diet_conflict": "Requested dish conflicts with dietary preferences: {}.",
    "insufficient_budget": "No dishes found within your budget of £{}.",
    "parsing_error": "Failed to parse natural language input. Please try again.",
    "agent_timeout": "Agent processing timed out. Please try again.",
}

SUCCESS_MESSAGES: Dict[str, str] = {
    "parsing_success": "Successfully parsed your request: {}",
    "agent_started": "Agent {} is analyzing...",
    "meal_selected": "Selected meal: {} (£{:.2f} for {} people)",
}

# ============================================================================
# Response Formatting
# ============================================================================

# Section headers for agent outputs
AGENT_OUTPUT_SECTIONS: List[str] = [
    "Planning Constraints",
    "Dish Analysis",
    "Cost and Nutrition Analysis",
    "Trade-off Explanation",
    "Shopping List",
    "Step-by-Step Cooking Execution Plan",
    "Robotics Handoff Summary",
]

# ============================================================================
# Feature Flags (for enabling/disabling features)
# ============================================================================

FEATURE_FLAGS: Dict[str, bool] = {
    "natural_language_parsing": True,
    "shopping_list_generation": True,
    "cost_breakdown": True,
    "nutrition_display": True,
    "multiple_suggestions": True,
    "allergen_checking": True,
    "price_negotiation": False,  # Future feature
    "voice_input": False,        # Future feature
    "image_upload": False,       # Future feature
}

# ============================================================================
# Utility Functions
# ============================================================================

def get_dietary_restrictions(filter_type: DietaryFilter) -> Set[str]:
    """Get the set of restricted ingredients for a dietary filter."""
    return DIETARY_RESTRICTIONS.get(filter_type, set())


def get_nutritional_targets(macro_focus: MacroFocus) -> Dict[str, float]:
    """Get nutritional targets for a given macro focus."""
    return NUTRITIONAL_TARGETS.get(macro_focus, NUTRITIONAL_TARGETS[MacroFocus.BALANCED])


def validate_budget(budget: float) -> bool:
    """Validate if budget is within acceptable range."""
    return MIN_BUDGET_GBP <= budget <= MAX_BUDGET_GBP


def validate_servings(servings: int) -> bool:
    """Validate if servings is within acceptable range."""
    return MIN_SERVINGS <= servings <= MAX_SERVINGS


def get_error_message(key: str, *args) -> str:
    """Get an error message with optional formatting."""
    message = ERROR_MESSAGES.get(key, "An unknown error occurred.")
    return message.format(*args) if args else message


def get_success_message(key: str, *args) -> str:
    """Get a success message with optional formatting."""
    message = SUCCESS_MESSAGES.get(key, "Operation completed successfully.")
    return message.format(*args) if args else message
