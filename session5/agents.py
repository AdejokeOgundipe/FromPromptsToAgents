"""
Multi-Agent System: Recipe Agent + Robotics Agent with A2A Communication.
=========================================================================
Session 5: The Challenge - Robotic Chef Platform

This module implements the Agent-to-Agent (A2A) pipeline:
1. Food Analysis Agent receives a dish name
2. It calls the Recipe MCP Server to analyse the dish
3. It creates a structured task specification for the Robotics Agent
4. The Robotics Agent uses the Robotics MCP Server to design a robot
5. The final robot specification is returned

The two agents communicate via a structured task specification - the output
of Agent 1 becomes the input to Agent 2. This is the core A2A pattern.

All LLM calls go through llm_client (local LLM service via requests).
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import llm_client
import config

# ============================================================================
# Logging Configuration
# ============================================================================

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create handlers
if not logger.handlers:
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

# Directory containing the MCP server scripts
SERVER_DIR = Path(__file__).parent
MAX_TOOL_OUTPUT_CHARS = 1200
MAX_HISTORY_CHARS = 5000
MAX_HISTORY_MESSAGES = 8
MAX_SYSTEM_PROMPT_CHARS = 2800
MAX_USER_MESSAGE_CHARS = 1800
MAX_TASK_SPEC_CHARS = 2000


# ---------------------------------------------------------------------------
# Natural Language Parser: Extract structured params from user input
# ---------------------------------------------------------------------------

def parse_natural_language_request(user_input: str) -> Dict[str, Any]:
    """
    Parse natural language input to extract meal planning parameters.
    
    Extracts structured parameters from free-form user input using regex patterns
    for budget, servings, dietary preferences, and macro focus.
    
    Examples:
      "I have £12 for two people. We need a high-protein meal. Design a robot to cook it."
      "Budget £8, 3 servings, vegan diet, carb-focused. Make jollof rice."
      
    Args:
        user_input: Free-form natural language request string.
    
    Returns:
        dict with keys:
        - budget_gbp: float or None (will default to config.DEFAULT_BUDGET_GBP)
        - servings: int or None (will default to config.DEFAULT_SERVINGS)
        - dietary_filter: str one of ["vegan", "gluten_free", "any"]
        - macro_focus: str one of ["protein", "carbohydrate", "balanced"]
        - dish_name: str or None (will prompt user if None)
        
    Raises:
        ValueError: If user_input is None or empty.
    """
    if not user_input or not isinstance(user_input, str):
        logger.error(f"Invalid user input: {type(user_input)}")
        raise ValueError("User input must be a non-empty string")
    
    logger.info(f"Parsing natural language input: {user_input[:100]}...")
    
    import re
    
    result = {
        "budget_gbp": None,
        "servings": None,
        "dietary_filter": "any",
        "macro_focus": "balanced",
        "dish_name": None,
    }
    
    try:
        # Extract budget: matches "£12", "12 pounds", "12 gbp", "$12" etc
        budget_patterns = [
            r'£\s*(\d+\.?\d*)',  # £12 or £ 12.50
            r'\$\s*(\d+\.?\d*)',  # $12
            r'(\d+\.?\d*)\s*(?:gbp|£|pounds?)',  # 12 gbp, 12 pounds
            r'budget[:\s]+(?:£|\$)?(\d+\.?\d*)',  # budget: £12
        ]
        for pattern in budget_patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                budget = float(match.group(1))
                if config.validate_budget(budget):
                    result["budget_gbp"] = budget
                    logger.debug(f"Extracted budget: £{budget}")
                else:
                    logger.warning(f"Budget {budget} out of valid range")
                break
        
        # Extract servings: matches "2 people", "for 3", "serves 4", "two servings" etc
        servings_patterns = [
            r'(?:for\s+)?(\w+)\s+(?:people|servings?|persons?)',  # "2 people", "three servings"
            r'serves?\s+(\w+)',  # "serves 4"
            r'(\w+)\s+(?:of\s+)?us',  # "4 of us"
        ]
        for pattern in servings_patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                number_word = match.group(1).lower()
                # Map words to numbers
                word_to_num = {
                    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
                }
                if number_word in word_to_num:
                    servings = word_to_num[number_word]
                    if config.validate_servings(servings):
                        result["servings"] = servings
                        logger.debug(f"Extracted servings (from word): {servings}")
                else:
                    try:
                        servings = int(number_word)
                        if config.validate_servings(servings):
                            result["servings"] = servings
                            logger.debug(f"Extracted servings (from number): {servings}")
                    except ValueError:
                        logger.debug(f"Could not parse servings: {number_word}")
                if result["servings"]:
                    break
        
        # Extract dietary preference
        if re.search(r'\bvegan\b', user_input, re.IGNORECASE):
            result["dietary_filter"] = "vegan"
            logger.debug("Dietary filter: vegan")
        elif re.search(r'(?:gluten[- ]?free|no gluten)', user_input, re.IGNORECASE):
            result["dietary_filter"] = "gluten_free"
            logger.debug("Dietary filter: gluten_free")
        elif re.search(r'\b(?:vegetarian|meat[- ]?free)\b', user_input, re.IGNORECASE):
            result["dietary_filter"] = "vegan"  # Vegetarian → vegan (more restrictive)
            logger.debug("Dietary filter: vegan (from vegetarian)")
        
        # Extract macro focus
        if re.search(r'(?:high[- ]?)?protein', user_input, re.IGNORECASE):
            result["macro_focus"] = "protein"
            logger.debug("Macro focus: protein")
        elif re.search(r'(?:high[- ]?)?carbs?|carbohydrates?', user_input, re.IGNORECASE):
            result["macro_focus"] = "carbohydrate"
            logger.debug("Macro focus: carbohydrate")
        elif re.search(r'balanced|mix', user_input, re.IGNORECASE):
            result["macro_focus"] = "balanced"
            logger.debug("Macro focus: balanced")
        
        # Extract dish name: look for patterns like "make", "cook", "prepare", "recipe for"
        dish_patterns = [
            r'(?:cook|make|prepare|recipe\s+for|design.*?(?:to\s+)?cook)\s+(?:a\s+)?(?:nice\s+)?([a-z\s]+?)(?:\.|,|!|$)',
            r'want\s+(?:a\s+)?([a-z\s]+?)(?:\.|,|!|$)',  # "I want pasta carbonara"
            r'dish\s+(?:preference|name)?\s*(?:is|:)\s*([a-z\s]+?)(?:\.|,|!|$)',
            r'([a-z\s]+?)\s+(?:please|ple?ase)$',  # At end of sentence
        ]
        for pattern in dish_patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                dish = match.group(1).strip()
                # Filter out common non-dish words
                bad_words = ['it', 'that', 'robot', 'design', 'a meal', 'recipe', 'food']
                if not any(x.lower() == dish.lower() for x in bad_words):
                    result["dish_name"] = dish
                    logger.debug(f"Extracted dish name: {dish}")
                    break
        
        logger.info(f"Parsing complete: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error parsing natural language request: {e}", exc_info=True)
        raise ValueError(f"Failed to parse request: {str(e)}")


# ---------------------------------------------------------------------------
# Core: Run an agent loop with an MCP server
# ---------------------------------------------------------------------------

async def run_agent_with_mcp(
    server_script: str,
    system_prompt: str,
    user_message: str,
    status_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Generic function to run an LLM agent loop connected to an MCP server.

    The agent will:
    1. Connect to the specified MCP server via stdio
    2. Discover available tools and convert them to a simple format
    3. Send the user message to the LLM with the tool definitions
    4. Execute any tool calls the LLM requests via the MCP session
    5. Feed tool results back to the LLM
    6. Repeat until the LLM produces a final text response (max 10 iterations)

    Args:
        server_script: Absolute or relative path to the MCP server Python file.
        system_prompt: The system prompt defining the agent's role and behaviour.
        user_message: The user's input message to the agent.
        status_callback: Optional callable(str) for real-time status updates.
                         Used by the Streamlit UI to show progress.

    Returns:
        The agent's final text response.
        
    Raises:
        Exception: If MCP server fails to start or agent loop fails.
    """

    def _status(msg: str):
        """Helper to call status callback if provided."""
        if status_callback:
            try:
                status_callback(msg)
            except Exception as e:
                logger.error(f"Status callback error: {e}")

    def _clamp_text(text: str, max_chars: int) -> str:
        """Clamp long text to keep prompts within model limits."""
        if not text:
            return ""
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n\n[truncated for model context limit]"

    def _compact_messages(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Keep context small enough for constrained local models.

        Preserves system + first user message and keeps the most recent
        exchanges within message and character budgets.
        """
        if len(history) <= 2:
            return history

        base = history[:2]
        tail = history[2:]

        # Keep only the most recent entries first.
        if len(tail) > MAX_HISTORY_MESSAGES:
            tail = tail[-MAX_HISTORY_MESSAGES:]

        # Enforce a rough character budget for tail messages.
        compact_tail: List[Dict[str, str]] = []
        current_chars = 0
        for msg in reversed(tail):
            msg_len = len(msg.get("content", ""))
            if current_chars + msg_len > MAX_HISTORY_CHARS:
                continue
            compact_tail.append(msg)
            current_chars += msg_len

        compact_tail.reverse()
        return base + compact_tail

    try:
        _status(f"Starting MCP server: {Path(server_script).name}")
        logger.info(f"Starting agent with MCP server: {server_script}")

        # Resolve the server script path
        server_path = str(Path(server_script).resolve())
        
        if not Path(server_path).exists():
            raise FileNotFoundError(f"Server script not found: {server_path}")

        # Set up MCP server connection via stdio
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[server_path],
        )

        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                # Initialise the MCP session
                await session.initialize()
                _status("MCP session initialised")
                logger.info("MCP session initialized successfully")

                # Discover tools and convert to simple format for llm_client
                tools_result = await session.list_tools()
                tools = [
                    {
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": t.inputSchema
                        if t.inputSchema
                        else {"type": "object", "properties": {}},
                    }
                    for t in tools_result.tools
                ]
                _status(
                    f"Discovered {len(tools)} tools: "
                    f"{', '.join(t['name'] for t in tools)}"
                )
                logger.info(f"Discovered {len(tools)} MCP tools")

                # Build initial conversation
                messages = [
                    {"role": "system", "content": _clamp_text(system_prompt, MAX_SYSTEM_PROMPT_CHARS)},
                    {"role": "user", "content": _clamp_text(user_message, MAX_USER_MESSAGE_CHARS)},
                ]

                # Agent loop (max 10 iterations to prevent runaway)
                last_content = ""
                for iteration in range(10):
                    try:
                        messages = _compact_messages(messages)
                        _status(f"LLM call (iteration {iteration + 1})")
                        logger.debug(f"Agent loop iteration {iteration + 1}")

                        response = llm_client.chat(messages, tools=tools)
                        logger.debug(f"LLM response: {len(response.get('content', ''))} chars")

                        # If the LLM made tool calls, execute them
                        if response.get("tool_calls"):
                            # Keep assistant context compact to avoid model context overflow.
                            messages.append(
                                {
                                    "role": "assistant",
                                    "content": response.get("content", "")
                                    or "Tool calls requested. Processing tool results.",
                                }
                            )

                            for tc in response["tool_calls"]:
                                fn_name = tc.get("name", "unknown")
                                fn_args = tc.get("arguments", {})

                                _status(f"Calling tool: {fn_name}")
                                logger.debug(f"Executing tool: {fn_name} with args: {fn_args}")

                                # Execute the tool via MCP
                                try:
                                    result = await session.call_tool(fn_name, fn_args)
                                    # Extract text content from the MCP result
                                    tool_output = ""
                                    if result.content:
                                        for content_block in result.content:
                                            if hasattr(content_block, "text"):
                                                tool_output += content_block.text
                                    _status(
                                        f"Tool {fn_name} returned {len(tool_output)} chars"
                                    )
                                    logger.debug(f"Tool {fn_name} output length: {len(tool_output)}")
                                except Exception as e:
                                    tool_output = json.dumps({"error": str(e)})
                                    logger.error(f"Tool {fn_name} execution failed: {e}", exc_info=True)
                                    _status(f"Tool {fn_name} error: {e}")

                                # Add tool result to conversation
                                if len(tool_output) > MAX_TOOL_OUTPUT_CHARS:
                                    tool_output = (
                                        tool_output[:MAX_TOOL_OUTPUT_CHARS]
                                        + "\n\n[truncated for context size safety]"
                                    )

                                messages.append(
                                    {
                                        "role": "tool",
                                        "name": fn_name,
                                        "content": tool_output,
                                    }
                                )
                        else:
                            # No tool calls -- the agent produced a final text response
                            _status("Agent produced final response")
                            logger.info("Agent produced final response")
                            return response.get("content") or ""

                        last_content = response.get("content") or ""

                    except Exception as e:
                        logger.error(f"Error in agent iteration {iteration + 1}: {e}", exc_info=True)
                        _status(f"Error in iteration {iteration + 1}: {e}")
                        raise

                # If we exhausted iterations, return whatever we have
                _status("Max iterations reached")
                logger.warning("Agent reached maximum iterations without final response")
                return (
                    last_content
                    or "Agent did not produce a final response within the iteration limit."
                )
    
    except FileNotFoundError as e:
        logger.error(f"Server file not found: {e}")
        _status(f"Error: Server file not found - {e}")
        raise
    except Exception as e:
        logger.error(f"Agent execution failed: {e}", exc_info=True)
        _status(f"Error: Agent execution failed - {e}")
        raise


# ---------------------------------------------------------------------------
# Agent 1: Food Analysis Agent
# ---------------------------------------------------------------------------

FOOD_ANALYSIS_SYSTEM_PROMPT = """\
You are the Food Analysis Agent, an expert culinary analyst and meal planner. Your role is to \
plan a budget-conscious meal, analyse its nutrition, and produce a detailed, structured task specification \
that a Robotics Design Agent can use to design an automated cooking robot.

When given a dish name, use the available tools to:
1. Analyse the dish fully (ingredients, steps, techniques)
2. Get the detailed cooking techniques
3. Get equipment specifications for key equipment
4. Get safety requirements
5. Use budget and nutrition tools to compare meal options before recommending one

INPUT PRIORITY RULES (must follow):
- Treat the user's requested dish/input as the PRIMARY target.
- If the requested dish is available, build the full analysis for that dish first.
- Only propose a fallback dish when the requested dish cannot satisfy hard constraints (budget/dietary).
- If fallback is used, clearly state why and keep the requested dish analysis visible.
- Never ignore the user's input and switch dishes silently.

Then synthesise all this information into a comprehensive TASK SPECIFICATION \
with the following clearly labelled sections:

FORMAT RULES (must follow):
- Use exactly the section headings shown below (## Heading)
- Keep each section concise and easy to scan
- Use short bullet points and numbered steps
- Put key numbers first (price, calories, protein, temperature, time)
- Avoid long paragraphs and avoid repeating the same information
- If data is unavailable, write "Not available" for that field
- Never include raw tool-call JSON, <tool_call> tags, or function argument dumps in the final answer

## Dish Overview
- Name, cuisine, difficulty, servings, total time

## Planning Constraints
- Budget, servings, dietary filter, macro priority

## Cost and Nutrition Analysis
- Estimated total price and price per person
- Protein, carbohydrates, calories, key vitamins
- Why this option is the best fit for the budget and nutrition target

## Physical Tasks Required
For each step, describe the physical action needed:
- Cutting/chopping (specify precision, force, dimensions)
- Stirring/mixing (specify speed, duration, force)
- Pouring/dispensing (specify volume, precision, temperature)
- Heating/temperature control (specify temperatures, durations, precision)
- Timing coordination (specify concurrent operations)

## Step-by-Step Cooking Execution Plan
Provide a numbered sequence the robot can follow from start to finish.
Each step must include:
- Step number and short action title
- Exact ingredients or components used
- Equipment needed
- Temperature if relevant
- Duration if relevant
- Expected result / checkpoint
- Any safety note for the robot

## Cooking Techniques with Precision Requirements
List each technique with:
- Temperature requirements (exact values in C)
- Duration requirements
- Precision level (critical/high/medium)
- Failure modes if done incorrectly

## Equipment to Operate
For each piece of equipment:
- What it is and how it must be operated
- Temperature ranges
- Physical interaction needed (knobs, handles, placement)

## Safety Requirements
- Temperature hazards and maximum temperatures
- Splash/splatter risks
- Timing-critical steps
- Food safety considerations

## Robotics Task Specification
A summary designed specifically for a Robotics Design Agent, listing:
- All manipulation tasks (with required degrees of freedom and force ranges)
- All sensing requirements (temperature, vision, force feedback)
- Workspace requirements (dimensions, stations)
- Speed and timing constraints
- Safety constraints for the robot

## Final Recommendation
- Selected dish or meal option
- One fallback option
- Shopping list with quantities for the requested servings

## Robotics Handoff Summary
- A concise, structured summary of the step-by-step execution plan
- Key manipulation tasks in order
- Critical safety and sensing requirements
- Any timing constraints that the Robotics Agent must preserve

Be thorough and specific - the Robotics Agent depends entirely on your analysis \
to design an appropriate robot. Include exact temperatures, durations, and force \
estimates wherever possible.
"""


async def run_food_analysis_agent(
    dish_name: str,
    budget_gbp: float,
    servings: int,
    dietary_filter: str,
    macro_focus: str,
    status_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Run Agent 1: Food Analysis Agent.

    Connects to the Recipe MCP Server and thoroughly analyses the specified dish,
    producing a structured task specification for the Robotics Agent.

    Args:
        dish_name: Name of the dish to analyse (e.g. 'pasta carbonara').
        budget_gbp: Budget in GBP for the meal.
        servings: Number of servings required.
        dietary_filter: Dietary filter ('vegan', 'gluten_free', 'any').
        macro_focus: Macro focus area ('protein', 'carbohydrate', 'balanced').
        status_callback: Optional callable(str) for real-time status updates.

    Returns:
        A detailed task specification string.
        
    Raises:
        ValueError: If parameters are invalid.
        Exception: If agent execution fails.
    """
    logger.info(f"Starting Food Analysis Agent for dish: {dish_name}")
    logger.info(f"  Budget: £{budget_gbp}, Servings: {servings}, Diet: {dietary_filter}, Macro: {macro_focus}")
    
    try:
        # Validate parameters
        if not dish_name or not isinstance(dish_name, str):
            raise ValueError("dish_name must be a non-empty string")
        if not config.validate_budget(budget_gbp):
            raise ValueError(f"Budget must be between £{config.MIN_BUDGET_GBP} and £{config.MAX_BUDGET_GBP}")
        if not config.validate_servings(servings):
            raise ValueError(f"Servings must be between {config.MIN_SERVINGS} and {config.MAX_SERVINGS}")
        
        server_script = str(SERVER_DIR / "recipe_mcp_server.py")
        user_message = (
            f"The user requested dish is '{dish_name}'. You must prioritise this exact input. "
            f"First analyse '{dish_name}' directly using the tools. "
            f"The meal must fit within a budget of GBP {budget_gbp:.2f}, serve {servings} people, "
            f"and respect the dietary filter '{dietary_filter}' with macro priority '{macro_focus}'. "
            f"Use get_price, get_nutrition, and fit_budget to evaluate feasibility and trade-offs. "
            f"Do not replace the requested dish unless constraints are impossible to satisfy. "
            f"If replacement is required, include: (1) why the requested dish fails constraints, "
            f"(2) the best fallback option, and (3) a clear label that fallback was used. "
            f"Explain the trade-offs between nutrition, cost, and cooking feasibility before producing the final task specification. "
            f"Return a detailed step-by-step cooking execution plan that a robot could follow directly, and then hand that plan off clearly to the Robotics Design Agent."
        )

        result = await run_agent_with_mcp(
            server_script=server_script,
            system_prompt=FOOD_ANALYSIS_SYSTEM_PROMPT,
            user_message=user_message,
            status_callback=status_callback,
        )
        
        logger.info(f"Food Analysis Agent completed successfully ({len(result)} chars)")
        return result
        
    except ValueError as e:
        logger.error(f"Validation error in Food Analysis Agent: {e}")
        raise
    except Exception as e:
        logger.error(f"Food Analysis Agent failed: {e}", exc_info=True)
        raise


# ---------------------------------------------------------------------------
# Agent 2: Robotics Designer Agent
# ---------------------------------------------------------------------------

ROBOTICS_DESIGN_SYSTEM_PROMPT = """\
You are the Robotics Design Agent, an expert in designing robotic systems for \
food preparation and cooking tasks. You receive a detailed task specification \
from the Food Analysis Agent and must design a complete robotic cooking platform.

Use the available tools to:
1. Search for suitable robot arms/platforms based on the task requirements
2. Find appropriate sensors for the required sensing capabilities
3. Find actuators and end-effectors for the required manipulation tasks
4. Get detailed specifications for each selected component
5. Use the recommendation tool for an initial platform suggestion

Then design a complete robotic system with these clearly labelled sections:

FORMAT RULES (must follow):
- Keep sections concise and easy to scan
- Use bullet points, short sentences, and clear tables where useful
- Put key technical values first (DoF, payload, range, speed, temperature limits)
- Never include raw tool-call JSON, <tool_call> tags, or function argument dumps in the final answer

## Robot Design Overview
- Robot type and form factor rationale
- Single-arm vs dual-arm justification
- Stationary vs mobile justification

## Selected Components
For each component, provide:
- Component ID and name
- Key specifications
- Why it was chosen for this specific dish

## Sensor Suite
For each sensor:
- Sensor ID and name
- What it monitors and why
- Mounting location recommendation

## Actuators and End-Effectors
For each actuator:
- Actuator ID and name
- What task it performs
- Key specifications relevant to the cooking task

## Motion and Control Requirements
- Degrees of freedom needed and why
- Speed requirements for time-critical operations
- Force control requirements
- Coordination between multiple operations

## Safety and Compliance
- How the robot handles high-temperature operations safely
- Human-robot interaction safety measures
- Food safety compliance
- Emergency stop scenarios

## Platform Summary Table
A clear summary table with all selected components, their IDs, and roles.

## Estimated Capabilities
- Which steps the robot can perform fully autonomously
- Which steps may need human oversight
- Overall autonomy percentage estimate

Be specific and reference actual component IDs from the database. Justify every \
selection based on the task specification you received.
"""


async def run_robotics_agent(
    task_specification: str,
    status_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Run Agent 2: Robotics Designer Agent.

    Connects to the Robotics MCP Server and designs a complete robotic platform
    based on the task specification from the Food Analysis Agent.

    Args:
        task_specification: The detailed task specification from Agent 1.
        status_callback: Optional callable(str) for real-time status updates.

    Returns:
        A detailed robot design specification string.
        
    Raises:
        ValueError: If task_specification is invalid.
        Exception: If agent execution fails.
    """
    logger.info("Starting Robotics Designer Agent")
    
    try:
        if not task_specification or not isinstance(task_specification, str):
            raise ValueError("task_specification must be a non-empty string")
        
        logger.debug(f"Task specification length: {len(task_specification)} chars")
        
        server_script = str(SERVER_DIR / "robotics_mcp_server.py")
        compact_spec = task_specification[:MAX_TASK_SPEC_CHARS]
        if len(task_specification) > MAX_TASK_SPEC_CHARS:
            compact_spec += "\n\n[task specification truncated for model context limit]"

        user_message = (
            f"Based on the following task specification from the Food Analysis Agent, "
            f"design a complete robotic cooking platform. Search the component databases "
            f"thoroughly and select the best components for each requirement.\n\n"
            f"--- TASK SPECIFICATION ---\n{compact_spec}\n--- END SPECIFICATION ---"
        )

        result = await run_agent_with_mcp(
            server_script=server_script,
            system_prompt=ROBOTICS_DESIGN_SYSTEM_PROMPT,
            user_message=user_message,
            status_callback=status_callback,
        )
        
        logger.info(f"Robotics Designer Agent completed successfully ({len(result)} chars)")
        return result
        
    except ValueError as e:
        logger.error(f"Validation error in Robotics Agent: {e}")
        raise
    except Exception as e:
        logger.error(f"Robotics Agent failed: {e}", exc_info=True)
        raise


# ---------------------------------------------------------------------------
# Pipeline: Full Robotic Chef Pipeline (A2A)
# ---------------------------------------------------------------------------

async def run_robotic_chef_pipeline(
    dish_name: str,
    budget_gbp: float,
    servings: int,
    dietary_filter: str,
    macro_focus: str,
    status_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, str]:
    """
    Run the full Robotic Chef A2A pipeline.

    This is the main entry point that orchestrates both agents:
    1. Runs the Food Analysis Agent to analyse the dish
    2. Passes the task specification to the Robotics Designer Agent
    3. Returns both outputs

    Args:
        dish_name: Name of the dish (e.g. 'pasta carbonara').
        budget_gbp: Budget in GBP for the meal.
        servings: Number of servings required.
        dietary_filter: Dietary filter ('vegan', 'gluten_free', 'any').
        macro_focus: Macro focus area ('protein', 'carbohydrate', 'balanced').
        status_callback: Optional callable(str) for real-time status updates.

    Returns:
        A dict with keys:
            - 'food_analysis': str - The Food Analysis Agent's output
            - 'robot_design': str - The Robotics Designer Agent's output
            
    Raises:
        ValueError: If parameters are invalid.
        Exception: If pipeline execution fails.
    """
    logger.info("="*60)
    logger.info(f"Starting Robotic Chef Pipeline")
    logger.info(f"  Dish: {dish_name}")
    logger.info(f"  Budget: £{budget_gbp}, Servings: {servings}")
    logger.info(f"  Diet: {dietary_filter}, Macro: {macro_focus}")
    logger.info("="*60)

    def _status(msg: str):
        if status_callback:
            try:
                status_callback(msg)
            except Exception as e:
                logger.error(f"Status callback error: {e}")

    try:
        # ---- Stage 1: Food Analysis Agent ----
        _status("=== Stage 1: Food Analysis Agent ===")
        logger.info("Stage 1: Running Food Analysis Agent")
        
        food_analysis = await run_food_analysis_agent(
            dish_name=dish_name,
            budget_gbp=budget_gbp,
            servings=servings,
            dietary_filter=dietary_filter,
            macro_focus=macro_focus,
            status_callback=status_callback,
        )
        _status("Food Analysis Agent complete")
        logger.info(f"Food Analysis Agent produced {len(food_analysis)} chars of output")

        # ---- Stage 2: Robotics Designer Agent ----
        _status("=== Stage 2: Robotics Designer Agent ===")
        logger.info("Stage 2: Running Robotics Designer Agent")
        
        robot_design = await run_robotics_agent(
            task_specification=food_analysis,
            status_callback=status_callback,
        )
        _status("Robotics Designer Agent complete")
        logger.info(f"Robotics Designer Agent produced {len(robot_design)} chars of output")

        logger.info("="*60)
        logger.info("Robotic Chef Pipeline completed successfully")
        logger.info("="*60)
        
        return {
            "food_analysis": food_analysis,
            "robot_design": robot_design,
        }
        
    except ValueError as e:
        logger.error(f"Validation error in pipeline: {e}")
        _status(f"Error: {e}")
        raise
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        _status(f"Error: Pipeline failed - {e}")
        raise


# ---------------------------------------------------------------------------
# CLI entry point (for testing without Streamlit)
# ---------------------------------------------------------------------------

async def _main():
    """Run the pipeline from the command line for testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Robotic Chef Pipeline - CLI")
    parser.add_argument(
        "dish",
        nargs="?",
        default="pasta carbonara",
        help="Name of the dish to analyse (default: pasta carbonara)",
    )
    parser.add_argument("--budget", type=float, default=15.0, help="Budget in GBP")
    parser.add_argument("--servings", type=int, default=2, help="Number of servings")
    parser.add_argument("--dietary-filter", default="vegan", help="Dietary filter (vegan, gluten_free, any)")
    parser.add_argument("--macro-focus", default="protein", help="Macro focus (protein, carbohydrate, balanced)")
    args = parser.parse_args()

    def print_status(msg: str):
        print(f"  [{msg}]")

    print(f"\nRobotic Chef Pipeline - Analysing: {args.dish}")
    print("=" * 60)

    result = await run_robotic_chef_pipeline(
        dish_name=args.dish,
        budget_gbp=args.budget,
        servings=args.servings,
        dietary_filter=args.dietary_filter,
        macro_focus=args.macro_focus,
        status_callback=print_status,
    )

    print("\n" + "=" * 60)
    print("FOOD ANALYSIS (Agent 1)")
    print("=" * 60)
    print(result["food_analysis"])

    print("\n" + "=" * 60)
    print("ROBOT DESIGN (Agent 2)")
    print("=" * 60)
    print(result["robot_design"])


if __name__ == "__main__":
    asyncio.run(_main())
