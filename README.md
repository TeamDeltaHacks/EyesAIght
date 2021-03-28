# EyesAIght

Revolutionizes the opthamology field using Machine Learning

## Dependencies
Due to GitHub's storage limit, we could not include one of the large trained model files in this repository. It can be downloaded [here](https://drive.google.com/file/d/1VhnAmoCXOwVYKVYtN_HhohBovc5Zk-v3/view?usp=sharing) and placed in the root directory of the repository.

Additionally, you may need to use `pip install` to install the dependencies on each line of `requirements.txt` (for example, `pip install flask==1.1.2`). This list is not comprehensive and does not include many of the modules, which will need to be installed as errors arise with running (see below section).

## Running
To run the code, open a terminal and navigate to the root directory of this repository. Then, run the following commands:
```bash
export FLASK_APP=hello.py
flask run
```

As import errors arise, you may need to install more modules by using `pip install MODULE_NAME_HERE`.

Finally, once the program runs without errors, navigate to `localhost:5000` in your browser. Enjoy!
