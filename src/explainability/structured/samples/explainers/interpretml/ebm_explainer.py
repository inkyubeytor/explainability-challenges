from interpret.blackbox import MorrisSensitivity
from sklearn.pipeline import Pipeline

from explainability.structured.core.structured_explainer import Path, \
    SKExplainer
from explainability.structured.core.structured_manipulator import \
    StructuredManipulator


class EBMExplainer(SKExplainer):
    """
    Uses explainable boosting machine global explanations to explain challenges.
    """

    def explain_global(self, trained_model: Pipeline,
                       sm: StructuredManipulator,
                       path: Path) -> None:
        model = trained_model.named_steps["model"]
        g = model.explain_global()
        fig = g.visualize()
        fig.write_image(str(path), format="png")
