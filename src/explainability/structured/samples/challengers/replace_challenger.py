from explainability.structured.core.structured_challenger import SKChallenger
from explainability.structured.core.structured_manipulator import \
    StructuredManipulator


class ReplaceChallenger(SKChallenger):
    """
    Challenger focusing on applying ad-hoc categorization to features.
    """

    def generate_challenges(self) -> None:
        """
        Generate multiple random value replacement challenges.
        """
        for i in range(2, 5):
            prop = 1 / i
            name = f"challenge_{prop}"
            sm = StructuredManipulator(self.df, self.label_column,
                                       self.random_state)
            sm.replace_random_values(proportion=prop)
            self.challenges[name] = sm
