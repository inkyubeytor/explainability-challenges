from explainability.structured.core.structured_challenger import SKChallenger
from explainability.structured.core.structured_manipulator import \
    StructuredManipulator
import random


class CategorizeChallenger(SKChallenger):
    """
    Challenger focusing on applying ad-hoc categorization to features.
    """

    def generate_challenges(self) -> None:
        """
        Generate multiple ad-hoc categorization challenges.
        """
        for i in range(2, 16):
            name = f"challenge_{i}"
            sm = StructuredManipulator(self.df, self.label_column)
            sm.categorize(num_bins=random.randint(2, 4))
            self.challenges[name] = sm
