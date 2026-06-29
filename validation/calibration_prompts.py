"""A few neutral calibration prompts to elicit realistic activation distributions.
Replace/extend with domain text relevant to the target deployment."""

CALIBRATION_PROMPTS = [
    "The history of computing began long before electronic machines existed.",
    "Photosynthesis converts light energy into chemical energy stored in glucose.",
    "In economics, supply and demand describe how prices form in a market.",
    "A neural network learns by adjusting weights to minimize a loss function.",
    "The Mediterranean climate is characterized by hot, dry summers and mild winters.",
    "Quantization reduces the numerical precision of a model to save memory and energy.",
    "She walked along the river at dawn, watching the mist rise over the water.",
    "The recipe calls for flour, eggs, butter, and a pinch of salt.",
]

# A longer passage for perplexity measurement (the 'does drift matter for the task?' check).
PERPLEXITY_TEXT = (
    "Language models predict the probability of the next token given the previous ones. "
    "Training adjusts billions of parameters so that frequent, sensible continuations get "
    "higher probability. At inference time, the model turns a prompt into a sequence of "
    "tokens, computes hidden representations through many transformer layers, and produces a "
    "distribution over the vocabulary. Efficient deployment on small devices requires reducing "
    "both the memory needed to store the weights and the energy spent moving them for each "
    "generated token, which is why low-bit and integer-only arithmetic have become important."
)
