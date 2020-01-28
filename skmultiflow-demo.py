########################################
from sklearn.linear_model import SGDClassifier
from skmultiflow.data import SEAGenerator
from skmultiflow.evaluation import EvaluatePrequential

# We will use the `SEA` stream generator
stream = SEAGenerator(classification_function=2, random_state=1)
# Prepare the stream for use
stream.prepare_for_use()
# Setup a classifier, in this case `Linear SVM` with `SGD` training*
classifier = SGDClassifier()
# Setup the evaluator, we will use prequential evaluation
eval = EvaluatePrequential(show_plot=True, max_samples=20000,
                           metrics=['accuracy', 'kappa', 'running_time', 'model_size'])
# Run the evaluation
eval.evaluate(stream=stream, model=classifier, model_names=['SVM-SGD']);
