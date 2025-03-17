class Constraints:
    def __init__(self):
        pass

    def is_feasible(self, solution):
        """
        Check if the given solution satisfies all constraints.
        This method should be overridden by specific constraint implementations.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def penalty(self, solution):
        """
        Calculate the penalty for violating constraints.
        This method should be overridden by specific constraint implementations.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")