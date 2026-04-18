"""
Robotic Chef Platform - Multi-Agent AI System
===============================================
Session 5: The Challenge - Agent-to-Agent (A2A) Integration

This Streamlit app integrates two AI agents:
- Agent 1: Food Analysis Agent (analyses dishes using Recipe MCP Server)
- Agent 2: Robotics Agent (designs robots using Robotics MCP Server)

All LLM calls go through llm_client (local LLM service via requests).

Run with:
    streamlit run app.py
"""

import streamlit as st
import asyncio
import os
import logging
import re
from dotenv import load_dotenv
from typing import Dict, Optional, Callable

from agents import run_robotic_chef_pipeline, parse_natural_language_request
import llm_client
import config

# ============================================================================
# Logging Configuration
# ============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Load Environment
# ============================================================================

load_dotenv()

# ============================================================================
# Streamlit Page Configuration
# ============================================================================

st.set_page_config(
    page_title="Robotic Chef Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# Sidebar Configuration
# ============================================================================

with st.sidebar:
    st.header("🤖 Robotic Chef Platform")
    
    st.markdown(
        """
        ### How It Works
        
        This platform demonstrates **Agent-to-Agent (A2A)** communication
        between two specialised AI agents:

        **Agent 1: Food Analysis**
        - Analyses dishes comprehensively
        - Evaluates budget & nutrition trade-offs
        - Produces detailed task specifications
        - Generates shopping lists

        **Agent 2: Robotics Designer**
        - Receives task specification from Agent 1
        - Designs custom robotic platforms
        - Selects optimal components
        - Produces robot design specifications

        The output of Agent 1 flows directly into Agent 2 -- this is
        the A2A pattern in action.
        """
    )

    st.divider()
    st.header("📚 Featured Dishes")
    st.markdown(
        """
        **Budget-Friendly (£5-7)**
        - Vegan Lentil Curry
        - Tofu Stir-Fry
        - Chickpea Pasta
        
        **Standard Options (£7-10)**
        - Pasta Carbonara
        - Pizza Margherita
        - Pad Thai
        - French Omelette

        **Premium Dishes (£10-12)**
        - Cheese Souffle
        - Sushi Rolls
        - Fish and Chips
        - Beef Stir-Fry
        
        **Special**
        - Chocolate Cake
        - Artisan Bread
        """
    )

    st.divider()
    
    # Configuration display
    with st.expander("⚙️ Configuration", expanded=False):
        st.markdown("**Budget Range**")
        st.text(f"Min: £{config.MIN_BUDGET_GBP} | Max: £{config.MAX_BUDGET_GBP}")
        
        st.markdown("**Servings Range**")
        st.text(f"Min: {config.MIN_SERVINGS} | Max: {config.MAX_SERVINGS}")
        
        st.markdown("**Features Enabled**")
        features = [k for k, v in config.FEATURE_FLAGS.items() if v]
        for feat in features:
            st.text(f"✓ {feat.replace('_', ' ').title()}")

    st.divider()
    st.caption(
        "**AI Workshop - Session 5: The Challenge**\n\n"
        "University of Hertfordshire"
    )

# ============================================================================
# Main Content
# ============================================================================

st.title("🤖 Robotic Chef Platform")
st.markdown("### Intelligent Meal Planning & Robot Design")
st.markdown(
    """
    Design custom meals and robots to cook them. Specify your budget, dietary preferences,
    and nutritional targets, and two AI agents will plan your meal and design a robot to make it.
    """
)

# ============================================================================
# Health Check
# ============================================================================

llm_url = os.getenv("LLM_SERVICE_URL", "http://localhost:8000")
llm_token = os.getenv("LLM_API_TOKEN", "")

if not llm_token or llm_token == "your-token-here":
    st.warning(
        "**⚠️ LLM API token not configured.** "
        "Please create a `.env` file with your LLM credentials:\n\n"
        "```\nLLM_SERVICE_URL=http://localhost:8000\nLLM_API_TOKEN=your-token\n```"
    )


# ============================================================================
# Helper Functions
# ============================================================================

def parse_and_display_cost_breakdown(budget: float, selected_dish: str, servings: int) -> Dict:
    """
    Helper to display cost breakdown section.
    Returns expected cost data structure.
    """
    estimated_per_person = budget / max(1, servings)
    return {
        "total_budget": budget,
        "estimated_total": budget * 0.85,  # Assume 85% of budget used
        "per_person": estimated_per_person,
        "servings": servings,
        "savings": budget * 0.15,
    }


def split_markdown_sections(markdown_text: str) -> Dict[str, str]:
    """Split markdown content by level-2 headings into named sections."""
    if not markdown_text or not isinstance(markdown_text, str):
        return {}

    matches = list(re.finditer(r"^##\s+(.+)$", markdown_text, flags=re.MULTILINE))
    if not matches:
        return {}

    sections: Dict[str, str] = {}
    for idx, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(markdown_text)
        content = markdown_text[start:end].strip()
        if content:
            sections[title] = content
    return sections


def sanitize_agent_output(text: str) -> str:
    """Remove tool-calling artifacts from agent text before rendering."""
    if not text or not isinstance(text, str):
        return ""

    cleaned = text

    # Remove explicit tool call tags if present
    cleaned = re.sub(r"<tool_call>\s*\{.*?\}\s*</tool_call>", "", cleaned, flags=re.DOTALL)

    # Remove bare JSON tool call lines/blocks commonly leaked by text-based tool calling
    cleaned = re.sub(
        r"\{\s*\"name\"\s*:\s*\"[^\"]+\"\s*,\s*\"arguments\"\s*:\s*\{.*?\}\s*\}",
        "",
        cleaned,
        flags=re.DOTALL,
    )

    # Normalize spacing after cleanup
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def render_food_analysis_response(food_analysis_text: str):
    """Render Agent 1 output in a clear, easy-to-read layout."""
    safe_text = sanitize_agent_output(food_analysis_text)
    sections = split_markdown_sections(safe_text)

    if not sections:
        st.markdown(safe_text)
        return

    preferred_order = [
        "Dish Overview",
        "Planning Constraints",
        "Cost and Nutrition Analysis",
        "Step-by-Step Cooking Execution Plan",
        "Final Recommendation",
        "Robotics Handoff Summary",
    ]

    quick_sections = [title for title in preferred_order if title in sections]
    if quick_sections:
        st.caption("Quick view")
        tabs = st.tabs(quick_sections)
        for i, title in enumerate(quick_sections):
            with tabs[i]:
                st.markdown(sections[title])

    with st.expander("Show full structured analysis", expanded=False):
        for title, content in sections.items():
            st.markdown(f"### {title}")
            st.markdown(content)


def display_agent_results(
    result: Dict[str, str],
    budget: float,
    servings: int,
    dietary_filter: str,
    macro_focus: str,
    dish_name: str,
):
    """Display agent results with improved formatting and sections."""
    food_analysis_text = sanitize_agent_output(result.get("food_analysis", ""))
    robot_design_text = sanitize_agent_output(result.get("robot_design", ""))
    
    st.divider()
    
    # Summary section
    st.subheader("📊 Meal Planning Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Budget", f"£{budget:.2f}")
    col2.metric("People", servings)
    col3.metric("Diet", dietary_filter.replace("_", " ").title())
    col4.metric("Focus", macro_focus.title())
    
    # Cost breakdown
    if config.FEATURE_FLAGS.get("cost_breakdown", True):
        st.subheader("💷 Estimated Cost Breakdown")
        cost_data = parse_and_display_cost_breakdown(budget, dish_name, servings)
        
        col_breakdown1, col_breakdown2, col_breakdown3 = st.columns(3)
        col_breakdown1.metric("Total Estimated Cost", f"£{cost_data['estimated_total']:.2f}")
        col_breakdown2.metric("Per Person", f"£{cost_data['per_person']:.2f}")
        col_breakdown3.metric("Budget Savings", f"£{cost_data['savings']:.2f}")
    
    # Nutrition information (if provided)
    if config.FEATURE_FLAGS.get("nutrition_display", True):
        st.subheader("🥗 Nutritional Information")
        st.info(
            "Nutritional targets are optimized for **"
            + macro_focus.replace("_", " ").title()
            + "** focus. Details provided in Agent 1 analysis below."
        )
    
    # Agent 1 Food Analysis Output
    st.subheader("🍽️ Agent 1: Food Analysis & Planning")
    render_food_analysis_response(food_analysis_text)
    
    # Agent 2 Robotics Design Output
    st.subheader("🤖 Agent 2: Robot Design Specification")
    with st.expander("⚙️ Complete Robot Design", expanded=True):
        st.markdown(robot_design_text)
    
    # Shopping List Toggle (for future integration)
    if config.FEATURE_FLAGS.get("shopping_list_generation", True):
        st.subheader("🛒 Shopping List")
        st.info(
            f"Shopping list for **{dish_name}** ({servings} servings) can be generated by Agent 1. "
            "View details in the Food Analysis section above."
        )


# ============================================================================
# Input Section
# ============================================================================

st.markdown("### 🎯 How would you like to input your meal request?")
input_mode = st.radio(
    "Choose your preferred input style:",
    options=["Natural Language", "Structured Input"],
    horizontal=True,
    label_visibility="collapsed",
)

# ============================================================================
# Natural Language Input Mode
# ============================================================================

if input_mode == "Natural Language":
    st.markdown("""
    **Free-form meal planning** - Describe what you want naturally and let the system parse it.
    
    Example: "I have £12 for two people. We need a high-protein meal without dairy."
    """)

    st.markdown("#### ⚙️ Quick Controls")
    col_nl_budget, col_nl_people = st.columns(2)
    with col_nl_budget:
        nl_budget_slider = st.slider(
            "Budget (£)",
            min_value=config.MIN_BUDGET_GBP,
            max_value=config.MAX_BUDGET_GBP,
            value=config.DEFAULT_BUDGET_GBP,
            step=config.BUDGET_STEP_GBP,
            key="nl_budget_slider",
            help="Default budget if not detected from text, or force via override"
        )
    with col_nl_people:
        nl_people_slider = st.slider(
            "Number of People",
            min_value=config.MIN_SERVINGS,
            max_value=config.MAX_SERVINGS,
            value=config.DEFAULT_SERVINGS,
            step=1,
            key="nl_people_slider",
            help="Default servings if not detected from text, or force via override"
        )

    use_slider_override = st.checkbox(
        "Use slider values for budget and people (ignore detected values)",
        value=False,
        key="nl_use_slider_override",
    )
    
    user_request = st.text_area(
        "Describe your meal request:",
        placeholder="I have £12 for two people. We need a high-protein meal. Design a robot to cook it.",
        label_visibility="collapsed",
        height=100,
    )
    
    col_nl1, col_nl2, col_nl3 = st.columns([4, 0.5, 1])
    with col_nl3:
        st.write("")
        run_button_nl = st.button("🚀 Plan Meal", type="primary", use_container_width=True, key="nl_button")
    
    if run_button_nl and user_request:
        # Parse natural language input
        try:
            st.info("🔍 Parsing your request...")
            parsed = parse_natural_language_request(user_request)
            
            # Apply defaults for missing parameters
            budget = parsed["budget_gbp"] if parsed["budget_gbp"] is not None else nl_budget_slider
            servings = parsed["servings"] if parsed["servings"] is not None else nl_people_slider

            if use_slider_override:
                budget = nl_budget_slider
                servings = nl_people_slider

            dietary_filter = parsed["dietary_filter"]
            macro_focus = parsed["macro_focus"]
            dish_name = parsed["dish_name"] or "signature meal"
            
            # Clamp values to valid ranges
            budget = max(config.MIN_BUDGET_GBP, min(budget, config.MAX_BUDGET_GBP))
            servings = max(config.MIN_SERVINGS, min(servings, config.MAX_SERVINGS))
            
            # Display what was understood
            with st.expander("📋 Parsed Request Details", expanded=False):
                st.success("✓ Request parsed successfully")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Budget", f"£{budget:.2f}")
                c2.metric("People", servings)
                c3.metric("Diet", dietary_filter)
                c4.metric("Focus", macro_focus)
                if parsed["dish_name"]:
                    st.success(f"**Dish Detected**: {dish_name}")
                else:
                    st.warning(f"**No specific dish found** - will use general planning")
            
            # Run the pipeline
            status_container = st.status(
                f"📝 Planning: **{dish_name}** for {servings} people (£{budget:.2f}, {macro_focus})",
                expanded=True
            )

            def status_callback_nl(msg: str):
                """Status callback for natural language mode."""
                with status_container:
                    st.text(msg)
            
            try:
                # Handle asyncio properly for Streamlit
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        result = pool.submit(
                            asyncio.run,
                            run_robotic_chef_pipeline(
                                dish_name,
                                budget,
                                servings,
                                dietary_filter,
                                macro_focus,
                                status_callback_nl,
                            ),
                        ).result()
                else:
                    result = asyncio.run(
                        run_robotic_chef_pipeline(
                            dish_name,
                            budget,
                            servings,
                            dietary_filter,
                            macro_focus,
                            status_callback_nl,
                        )
                    )

                status_container.update(
                    label="✅ Pipeline complete!",
                    state="complete",
                    expanded=False
                )
                
                # Display results
                display_agent_results(result, budget, servings, dietary_filter, macro_focus, dish_name)

            except Exception as e:
                status_container.update(label="❌ Pipeline failed", state="error")
                st.error(f"**Error during pipeline execution:** {e}")
                with st.expander("📌 Error Details"):
                    st.exception(e)

        except ValueError as e:
            st.error(f"**Failed to parse request:** {e}")
            st.info("Try rephrasing your request more explicitly, e.g., 'I have £15 for 2 people. Make pasta.'")

    elif run_button_nl and not user_request:
        st.warning("Please enter a meal request to get started.")

# ============================================================================
# Structured Input Mode
# ============================================================================

else:  # Structured Input
    st.markdown("""
    **Guided form input** - Specify your preferences using sliders and dropdowns.
    """)
    
    st.markdown("#### ⚙️ Configure Your Meal")
    col_s1, col_s2, col_s3, col_s4 = st.columns([1, 1, 1, 1])

    with col_s1:
        budget = st.slider(
            "Budget (£)",
            min_value=config.MIN_BUDGET_GBP,
            max_value=config.MAX_BUDGET_GBP,
            value=config.DEFAULT_BUDGET_GBP,
            step=config.BUDGET_STEP_GBP,
            help="Total budget for all ingredients"
        )

    with col_s2:
        servings = st.slider(
            "Number of People",
            min_value=config.MIN_SERVINGS,
            max_value=config.MAX_SERVINGS,
            value=config.DEFAULT_SERVINGS,
            step=1,
            help="How many servings to prepare"
        )

    with col_s3:
        dietary_filter = st.selectbox(
            "Dietary Preference",
            ["any", "vegan", "gluten_free"],
            help="Filter by dietary requirements"
        )

    with col_s4:
        macro_focus = st.selectbox(
            "Nutrition Focus",
            ["balanced", "protein", "carbohydrate"],
            help="Optimize meal composition"
        )

    st.markdown("#### 🍽️ Meal Preference")
    dish_name = st.text_input(
        "Preferred Dish (optional)",
        placeholder="e.g. lentil curry, tofu stir-fry, chickpea pasta, or leave blank for AI suggestion...",
        help="Specific dish name, or leave empty for AI recommendation"
    )

    col_s_btn1, col_s_btn2 = st.columns([4, 1])
    with col_s_btn2:
        run_button_s = st.button("🚀 Plan Meal", type="primary", use_container_width=True, key="struct_button")

    # Pipeline execution for structured mode
    if run_button_s and (dish_name or True):  # Allow empty dish_name
        if not dish_name:
            dish_name = "chef's recommendation"
        
        status_container = st.status(
            f"📝 Planning: **{dish_name}** for {servings} people (£{budget:.2f})",
            expanded=True
        )

        def status_callback_s(msg: str):
            """Status callback for structured input mode."""
            with status_container:
                st.text(msg)

        try:
            # Handle asyncio properly for Streamlit
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = pool.submit(
                        asyncio.run,
                        run_robotic_chef_pipeline(
                            dish_name,
                            budget,
                            servings,
                            dietary_filter,
                            macro_focus,
                            status_callback_s,
                        ),
                    ).result()
            else:
                result = asyncio.run(
                    run_robotic_chef_pipeline(
                        dish_name,
                        budget,
                        servings,
                        dietary_filter,
                        macro_focus,
                        status_callback_s,
                    )
                )

            status_container.update(
                label="✅ Pipeline complete!",
                state="complete",
                expanded=False
            )
            
            # Display results
            display_agent_results(result, budget, servings, dietary_filter, macro_focus, dish_name)

        except Exception as e:
            status_container.update(label="❌ Pipeline failed", state="error")
            st.error(f"**Error during pipeline execution:** {e}")
            with st.expander("📌 Error Details"):
                st.exception(e)

