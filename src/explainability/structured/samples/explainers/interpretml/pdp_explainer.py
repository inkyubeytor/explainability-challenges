import numpy as np
from interpret.blackbox import PartialDependence
from sklearn.pipeline import Pipeline

from explainability.structured.core.structured_explainer import Path, \
    SKFeatureExplainer
from explainability.structured.core.structured_manipulator import \
    StructuredManipulator


class PDPExplainer(SKFeatureExplainer):
    """
    Uses partial dependence global explanations to explain challenges.
    """

    def explain_global(self, trained_model: Pipeline,
                       sm: StructuredManipulator,
                       feature: str,
                       path: Path) -> None:
        preprocessor = trained_model.named_steps["preprocessor"]

        x, _, _, _ = sm.train_test_split()

        x = preprocessor.transform(x)
        feature_names = preprocessor.get_feature_names_out()

        if feature not in feature_names:
            raise ValueError(f"feature {feature} not found."
                             f"Did you check its name after encoding?")
        feat_idx = np.where(feature_names == feature)[0].item()

        pdp = PartialDependence(trained_model.named_steps["model"],
                                x,
                                feature_names=feature_names)
        g = pdp.explain_global()
        fig = g.visualize(feat_idx)
        fig.write_image(str(path), format="png")
