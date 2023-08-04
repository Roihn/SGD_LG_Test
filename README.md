# SGD Predictor

LG AI research pre-coding test

## 💻 Get Started

```bash
# Clone Repository
git clone https://github.com/roihn/SGD_LG_Test.git
cd SGD_LG_Test

# Setup Environment
conda create --name sgd python=3.10
conda activate sgd
pip install -r requirements.txt
```

To run the tests, you need to download the checkpoints [here](https://drive.google.com/file/d/1giQwwn0jSTx2wc39miu5KP5S05heaoyq/view?usp=sharing)

After downloading the checkpoints, you need to unzip the `models.zip` file and place the following three folders in the `./models` folder. 

The file structure should look like this:

```bash
.
├── data
├── legacy
├── models
│   ├── model_act_100
│   ├── model_slot_100
│   └── model_value_100
├── src
...
```
where `model_act_100`, `model_slot_100`, and `model_value_100` are the three folders that you need to place in the `./models` folder.

You do not need to 

## 📝 Usage

```bash
python src/nlu.py <utterance>
```

For example:
```bash
(sgd) python src/nlu.py "I am feeling hungry so I would like to find a place to eat."
(act=INFORM_INTENT, slot=intent, value=FindRestaurants)
```

## 📝 Train
To train the model, you need to first separately train the three models for act, slot, and value prediction. 

```bash
python main.py --model act
python main.py --model slot
python main.py --model value
```

Then we can apply the evaluation script to evaluate the performance of the merged models.

```bash
python src/eval_limited.py
```

### Varying supervision
To train the model with varying supervision, you can simply pass the `--proportion` argument to the `main.py` script.

```bash
python main.py --model act --proportion 10
python main.py --model slot --proportion 10
python main.py --model value --proportion 10
```

and the related evaluation: 
```bash
python src/eval_limited.py --proportion 10 # for 10% supervision
```

Here is the list of proportions that is supported: [10, 30, 50, 100].

*Since there is no seed specified for training, the reproduced result may slightly differ.