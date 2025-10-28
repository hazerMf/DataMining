## Google Colab

```python
# Save và download
import joblib
from google.colab import files

# Save model
joblib.dump(rf, 'random_forest.joblib')

# Download
files.download('random_forest.joblib')

# Upload lên local: E:/SourceCode/DataMining/BE/models/random_forest/
```
