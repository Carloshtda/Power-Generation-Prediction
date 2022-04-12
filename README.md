# Power-Generation-Prediction
Analysis of Neural Network Models for Photovoltaic Power Generation taking into account periods of 2 hours fto predict the next 5-minutes.

## Results

### Baseline 1
Prediction is the power generated at the same time of the previous day.

| MAE | RMSE | STD |
| --------------- | --------------- | --------------- |
| 359.972 W | 779.223 W | 779.222 W |]

### Baseline 2
Prediction is *mean* of the power generated at the same time of the previous 10 days.

| MAE | RMSE | STD |
| --------------- | --------------- | --------------- |
| 313.928 W | 614.473 W | 614.470 W |]

### Linear Regression 

| MAE | RMSE | STD |
| --------------- | --------------- | --------------- |
| 126.586 W | 298.476 W | 298.474 W |]

### Single Hidden Layer MLP

|  | MAE | RMSE | STD |
| --------------- | --------------- | --------------- | --------------- |
| Training | 111.054 W | 328.128 W | 327.502 W |
| Validation | 111.489 W | 235.701 W | 234.881 W |
| Test | 106.121 W | 297.771 W | 297.643 W |]

### Optimized MLP

|  | MAE | RMSE | STD |
| --------------- | --------------- | --------------- | --------------- |
| Training | 107.659 W | 322.565 W | 321.847 W
| Validation | 107.305 W | 231.406 W | 230.470 W
| Test | 104.300 W | 307.345 W | 307.220 W]

### Single Hidden Layer fully conected RNN

|  | MAE | RMSE | STD |
| --------------- | --------------- | --------------- | --------------- |
| Training | 116.869 W | 342.489 W | 342.148 W
| Validation | 116.282 W | 245.317 W | 245.109 W
| Test | 109.304 W | 303.073 W | 302.565 W]

### Optimized fully conected RNN

|  | MAE | RMSE | STD |
| --------------- | --------------- | --------------- | --------------- |
| Training | 113.442 W | 332.453 W | 331.984 W |
| Validation | 112.902 W | 238.474 W | 238.104 W |
| Test | 103.077 W | 294.561 W | 294.109 W |]

### Single Hidden Layer LSTM

|  | MAE | RMSE | STD |
| --------------- | --------------- | --------------- | --------------- |
| Training | 115.469 W | 337.309 W | 336.946 W |
| Validation | 114.908 W | 241.901 W | 241.726 W |
| Test | 108.073 W | 303.225 W | 303.162 W |]

### Optimized LSTM

|  | MAE | RMSE | STD |
| --------------- | --------------- | --------------- | --------------- |
| Training | 114.731 W | 331.269 W | 332.046 W |
| Validation | 113.902 W | 241.901 W | 241.726 W |
| Test | 106.809 W | 299.220 W | 299.047 W |]

### Transformer-decoder

|  | MAE | RMSE | STD |
| --------------- | --------------- | --------------- | --------------- |
| Training | 163.972 W | 372.677 W | 369.463 W |
| Validation | 115.710 W | 168.729 W | 162.940 W |
| Test | 112.554 W | 148.695 W | 154.810 W |]

* Based on: https://github.com/nklingen/Transformer-Time-Series-Forecasting

