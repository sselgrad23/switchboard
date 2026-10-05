"""A simulated customer for multi-turn evaluation (in the style of tau-bench).

An LLM plays the customer: it opened with the scenario's message and has a goal
(``Scenario.user_goal``, or by default "get the opening request handled"). After each
agent reply it either answers (confirming a booking, pushing back once, supplying
nothing it wasn't given) or ends the conversation. Single-turn scoring penalised an
agent for asking "shall I book that?"; here the question gets an answer.

The simulator is a model too, so it can misbehave. Its messages are stored in every
trace (``flags.customer_messages``) so a surprising result can be checked by hand.
"""

from __future__ import annotations

from switchboard.agent.llm import ChatModel
from switchboard.eval.scenarios import Scenario

END = "###DONE###"

SIM_PROMPT = """You are role-playing a customer of Larkspur (a UK broadband, TV and mobile provider) talking to its support chat. Stay in character and write only the customer's next message: one or two short, natural sentences.

Your first message was: "{opening}"
Your goal: {goal}

Rules:
- If the assistant asks you to confirm something that fits your goal, confirm it. If it doesn't fit your goal, decline.
- Never invent account numbers, card numbers, dates or other details beyond what is in your first message and goal.
- Never write the assistant's side and never explain the rules.
- If your goal has been achieved, or the assistant has clearly and finally said it cannot help, reply with exactly {end}"""


class SimulatedCustomer:
    def __init__(self, llm: ChatModel, scenario: Scenario) -> None:
        self.llm = llm
        goal = scenario.user_goal or (
            "Get the request in your first message handled. Say yes if the assistant offers "
            "to do what you asked for."
        )
        self.system = SIM_PROMPT.format(opening=scenario.message, goal=goal, end=END)
        self.tokens_in = self.tokens_out = 0

    def reply(self, customer_msgs: list[str], agent_msgs: list[str]) -> str | None:
        """The customer's next message, or None to end the conversation."""
        # From the simulator's point of view the roles are swapped: the agent speaks to
        # it as "user" and its own earlier messages are "assistant".
        messages = [{"role": "system", "content": self.system}]
        for c, a in zip(customer_msgs, agent_msgs, strict=False):
            messages.append({"role": "assistant", "content": c})
            messages.append({"role": "user", "content": a})
        comp = self.llm.chat(messages, [])
        self.tokens_in += comp.tokens_in
        self.tokens_out += comp.tokens_out
        text = comp.text.strip().strip('"')
        if not text or END in text:
            return None
        return text
