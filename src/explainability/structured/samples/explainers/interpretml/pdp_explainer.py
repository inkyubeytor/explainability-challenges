from interpret.blackbox import PartialDependence
from sklearn.pipeline import Pipeline

from explainability.structured.core.structured_explainer import Path, \
    SKFeatureExplainer
from explainability.structured.core.structured_manipulator import \
    StructuredManipulator


class PDPExplainer(SKFeatureExplainer):
    """
    Uses Morris sensitivity global explanations to explain challenges.
    """

    def explain_global(self, trained_model: Pipeline,
                       sm: StructuredManipulator,
                       feature: str,
                       path: Path) -> None:
        encoder = trained_model.named_steps["encoder"]
        scaler = trained_model.named_steps["scaler"]

        x, _, _, _ = sm.train_test_split()

        x = encoder.transform(x)
        feature_names = [col.split("__")[1]
                         for col in encoder.get_feature_names_out()]
        x = scaler.transform(x)

        if feature not in feature_names:
            raise ValueError(f"feature {feature} not found."
                             f"Did you check its name after encoding?")
        feat_idx = feature_names.index(feature)

        pdp = PartialDependence(trained_model.named_steps["model"],
                                x,
                                feature_names=feature_names)
        g = pdp.explain_global()
        fig = g.visualize(feat_idx)
        fig.write_image(str(path), format="png")
