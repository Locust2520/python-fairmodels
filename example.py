import pandas as pd
import numpy as np
from fairmodels import *
from fairmodels.plotnine import *

# ~~~~~~~~~~~~~~~~~~~ loading the german dataset ~~~~~~~~~~~~~~~~~~~ #

german = pd.read_csv("test/german.csv")
y = (german.Risk == "good").astype(np.float)

# ~~~~~~~~~~~~~~~~~~~ loading models predictions ~~~~~~~~~~~~~~~~~~~ #

predictions = pd.read_csv("test/pred.csv")
lm = ModelProb(preds=predictions["lm"], threshold=0.5, name="Linear Model")
rf = ModelProb(preds=predictions["rf"], threshold=0.5, name="Random Forest")

# ~~~~~~~~~~~~~~~~~~~~~~~~ fairness checking ~~~~~~~~~~~~~~~~~~~~~~~ #

fobject = FairnessObject(
    model_probs=[lm, rf],
    y=y,
    protected=german.Sex,
    privileged="male"
)
plt = fobject.plot()
plt += theme_minimal()  # changing the theme
plt.show()
plt.save("graphic.png")
