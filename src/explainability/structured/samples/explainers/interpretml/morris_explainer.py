from interpret.blackbox import MorrisSensitivity
from sklearn.pipeline import Pipeline

from explainability.structured.core.structured_explainer import Path, \
    SKExplainer
from explainability.structured.core.structured_manipulator import \
    StructuredManipulator


class MorrisExplainer(SKExplainer):
    """
    Uses Morris sensitivity global explanations to explain challenges.
    """

    def explain_global(self, trained_model: Pipeline,
                       sm: StructuredManipulator,
                       path: Path) -> None:
        encoder = trained_model.named_steps["encoder"]
        scaler = trained_model.named_steps["scaler"]

        x, _, _, _ = sm.train_test_split()

        x = encoder.transform(x)
        feature_names = [col.split("__")[1]
                         for col in encoder.get_feature_names_out()]
        x = scaler.transform(x)

        msa = MorrisSensitivity(trained_model.named_steps["model"],
                                x,
                                feature_names=feature_names)
        g = msa.explain_global()
        fig = g.visualize()
        fig.write_image(str(path), format="png")
