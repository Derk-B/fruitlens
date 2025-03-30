# FruitLens

See [report.pdf](report.pdf) for documentation.

## Fruit image training instructions
Download the 100x100 fruit images [here](https://www.kaggle.com/datasets/moltean/fruits). 

### Running the training algorithm
Put the directory in the root of this project. Then run the `convert` function from `Convert.lhs`, or first build the project with `make build`, and then run the training algorithm with `stack exec convert-exe`.

### Running the classification algorithm
Run the `startServer` function from `API.lhs`, or first build the project with `make build`, and then run the server with `stack exec fruitlens-exe`.