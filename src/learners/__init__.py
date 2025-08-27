from .independent_ppo_learner import IndependentPPOLearner
from .independent_ppo_learner_srmt import IndependentPPOLearnerSRMT
from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .ppo_learner import PPOLearner
from .centralized_ppo_learner import CentralPPOLearner
from .trust_region_learner import TrustRegionLearner
from .joint_learner import JointLearner
from .trust_region_learner_srmt import TrustRegionLearnerSRMT

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["ppo_learner"] = PPOLearner
REGISTRY["central_ppo_learner"] = CentralPPOLearner
REGISTRY["trust_region_learner"] = TrustRegionLearner
REGISTRY["trust_region_learner_srmt"] = TrustRegionLearnerSRMT
REGISTRY["joint_learner"] = JointLearner
REGISTRY['independent_ppo_learner'] = IndependentPPOLearner
REGISTRY['independent_ppo_learner_srmt'] = IndependentPPOLearnerSRMT
