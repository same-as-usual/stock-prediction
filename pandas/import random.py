import random

# Define possible actions
ACTIONS = ["Move Left", "Move Right", "Suck Dirt"]

# Define State class
class State:
    def __init__(self, vacuum_pos, room_state, parent=None, action=None):
        self.vacuum_pos = vacuum_pos
        self.room_state = room_state
        self.parent = parent
        self.action = action

    def is_goal(self):
        return self.room_state["Left"] == "Clean" and self.room_state["Right"] == "Clean"

    def get_children(self):
        children = []

        # If current room is dirty, suck the dirt
        if self.room_state[self.vacuum_pos] == "Dirty":
            new_state = self.room_state.copy()
            new_state[self.vacuum_pos] = "Clean"
            children.append(State(self.vacuum_pos, new_state, self, "Suck Dirt"))

        # Move Left
        if self.vacuum_pos == "Right":
            children.append(State("Left", self.room_state.copy(), self, "Move Left"))

        # Move Right
        if self.vacuum_pos == "Left":
            children.append(State("Right", self.room_state.copy(), self, "Move Right"))

        return children

# Main Vacuum Cleaner Logic
class VacuumCleaner:
    def __init__(self):
        self.environment = {"Left": random.choice(["Clean", "Dirty"]),
                            "Right": random.choice(["Clean", "Dirty"])}
        self.vacuum_pos = random.choice(["Left", "Right"])

    def clean(self):
        print("Initial State:", self.environment, "| Vacuum at:", self.vacuum_pos)
        actions = []

        while "Dirty" in self.environment.values():
            if self.environment[self.vacuum_pos] == "Dirty":
                self.environment[self.vacuum_pos] = "Clean"
                actions.append(f"Sucked dirt in {self.vacuum_pos}")

            if self.vacuum_pos == "Left" and self.environment["Right"] == "Dirty":
                self.vacuum_pos = "Right"
                actions.append("Moved Right")
            elif self.vacuum_pos == "Right" and self.environment["Left"] == "Dirty":
                self.vacuum_pos = "Left"
                actions.append("Moved Left")

        print("Final State:", self.environment)
        print("Actions Taken:", actions)

# Run the vacuum cleaner simulation
vacuum = VacuumCleaner()
vacuum.clean()
