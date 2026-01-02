# Football (Soccer) Market Value Predictor

## Can a model approximate Transfermarkt's human-assigned market price for football players?

## Description
Market value in football is an estimated transfer price that reflects a player's age, potential, and market demand. The valuation drivers of market value include, but are not limited to, age, competition's prestige, team status, minutes played, goals, assists, cards, position, and contract length. Market value is also heavily influenced by broader market conditions as well as the player's past market value. A player's market price is important because it functions as a standard for budgeting, transfers, and salary negotiations. It is a key metric for clubs, agents, and scouts for decision-making, as well as for economists studying the market. This project answers whether a machine learning model can replicate Transfermarkt's manually assessed market values using data from historical players. It also looks at how close the predictions are compared to Transfermarkt values and where the model struggles most. The model is trained up to 2020 and makes predictions for seasons starting in 2021 onward.

This project is meant for learning and analysis, not for real-world transfer valuation.
## Features
Most features are from actual player statistics or metadata such as age, positional prime, and footedness. Some might be human-assigned values, but are shifted.

The features have been split into **two main categories**:
### 1. Objective Features
**a. Player performance per season**
- Goals, assists, goal contributions, minutes, clean sheets, goals conceded

**b. Career totals and averages**
- Career stats (totals of goals, assists, etc.) and per-season averages

**c. Performance trends**
- Rolling/weighted contributions, changes vs last season, position-normalized stats

**d. Age and experience**
- Age, age squared, years of experience, prime-age indicators

**e. Team-level performance**
- Team total goals, team average goals, team average goals per player

### 2. Human-influenced Features (human-assigned values) 
**a. Past market values**
- Peak value previous season, team/competition average, and median value

**b. Contract-related** 
- Contract length, contract remaining ratio, contract pressure score

## Dataset
**Source:** Public Kaggle dataset compiled using Transfermarkt data (CC0 1.0).

**Overview:**  
- 90,000+ players worldwide
- 900,000+ market valuations
- Player performance stats
- Contract information

**Time Span & Usage:** Earlier seasons (up to 2020) are used for training, while later seasons (starting 2021) are used to test the model's predictions.

**Target Variable:** Each player's market value, as estimated by Transfermarkt.

**Nuance:** Some features include past human-assigned values from Transfermarkt, which are shifted to avoid leakage.

## Model & Training
The model uses past player stats and features to predict value.

- **Train Data:** Seasons up to 2020

- **Test Data:** Seasons starting 2021

- **Model:** LightGBM with hyperparameter tuning

- **Target:** Log-transformed player market value
## Results & Evaluation
The model gives player market values that are fairly close to the values from Transfermarkt.

**Core measures:**
- **LightGBM (LGB) model:**
    - MAE (mean absolute error): €545,033
    - R²: 0.75

- **HistGradientBoosting (HGB) model (not used)**
    - MAE: €576,550
    - R²: 0.72

Errors tend to be larger with really high-valued players, likely because PR and hype play a big role in market value. It is also less accurate with younger players who have limited statistics, as well as players in smaller leagues.
## Project Structure
<strong>data/

models/

notebooks/

scripts/

src/

README.md </strong>
## How to Run
1. **Clone the repository**
```bash
git clone https://github.com/yss1n-0/football-soccer-player-market-value-predictor

cd football-soccer-player-market-value-predictor
```
2. **Set up a Python virtual environment**
```bash
python -m venv .venv
# Activate it
# Windows (Git bash / CMD):
source .venv/Scripts/activate

# Mac/Linux
source .venv/bin/activate
```
3. **Install dependencies**
```bash
pip install -r requirements.txt
```
4. **Preprocess and merge raw datasets (Optional)**
```bash
python scripts/preprocess_all.py
```
5. **Generate features (Optional)**
```bash
python src/feature_engineer.py
```
6. **Train the model**
```bash
python src/train_model.py
```
7. **Generate predictions**
```bash
python src/predict_model.py
```
8. **Plot results and evaluate errors**
```bash
python src/plot_results.py
```
## Limitations
As previously mentioned, some features reflect past human judgment, but the model is still being tested on new seasons to assess errors and well-predicted values. Although market value is supposed to reflect transfer fees, exact numbers often differ due to complex negotiations and situations. The dataset is static, so the model cannot account for new changes, which I intend to address in the future.
## Future Improvements
- Experiment with more player statistics and information, for example, injuries.
- Predict the likelihood of transfer.
- Add top transfer targets.
- Build a web app to search players and display their predicted vs actual market value.
## Author
Yassin Abdelghany – [yss1n-0](https://github.com/yss1n-0)
## License
MIT License – See the [LICENSE](LICENSE) file for details.