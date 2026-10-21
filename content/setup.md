# Setup

[Python][python] is a popular language for scientific computing, and a frequent choice for machine learning as well.

This workshop will use a Python environment containing **JupyterLab** and the necessary dependencies as the primary interface for all practical sessions. You are encouraged to follow the [local installation](#local-installation) option. [Google Colab](#colab-fallback) will be available as a fallback option if needed.

Please set up your Python environment at least a day in advance of the workshop. 
If you encounter problems with the installation procedure, attend the pre-workshop setup session for assistance, so that you are ready to go as soon as the workshop begins.

(local-installation)=
## Local installation

:::{important}
You have different options:

- Use [pip](#pip) and virtual environments: If you have `git` in your terminal and are
  comfortable using virtual environments and have a working Python interpreter
  (at least version 3.10 or above), or

- Use [Miniforge](#miniforge) / [pixi](#pixi) and conda environments: Recommended for Windows users, 
  and users who don't have tools like a Python interpreter, `git` in their devices.

Once you have done either, proceed to getting the [notebooks](#notebooks). Finally download the [datasets](#datasets)
:::

(pip)=
### Using pip

Open a terminal (Mac/Linux) or Command Prompt (Windows) and run the following commands.

1. Create a [virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#create-and-use-virtual-environments) called `workshop`:

`````{tabs}

````{group-tab} On Linux/macOS

```shell
python3 -m venv workshop
```

````

````{group-tab} On Windows

:::{warning}
For Windows consider using [Miniforge](#miniforge) or [pixi](#pixi) instead
:::

```shell
py -m venv workshop
```

````

`````

2. Activate the newly created virtual environment:

`````{tabs}

````{group-tab} On Linux/macOS

```shell
source workshop/bin/activate
```

````

````{group-tab} On Windows

```shell
workshop\Scripts\activate
```

````

`````

Remember that you need to activate your environment every time you restart your terminal!

3. Install the required packages:

Once you have activated your environment, you can install all the required packages for this workshop using the following command:

```shell
pip install -r https://raw.githubusercontent.com/sweden-ai-factory/xai-for-life-science/refs/heads/main/requirements.txt
```

(miniforge)=
### Using Miniforge

In this page, we give indications about how to install Miniforge and use conda/mamba.

1. The first step is to install [Miniforge][miniforge], which is a modified version of
_Miniconda_ using by default the
community driven [conda-forge] channel.

`````{tabs}

````{group-tab} On Linux/macOS

Download the installer using curl or wget or your favorite program. For eg:

```sh
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
```

or

```sh
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
```

Execute the installer and initialize conda:

```sh
bash Miniforge3-$(uname)-$(uname -m).sh -b
$HOME/miniforge3/bin/conda init --all
```
````

````{group-tab} On Windows
On Windows, download
[the Windows installer](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe)
and follow the instructions given in <https://github.com/conda-forge/miniforge>.

Open an Anaconda Prompt and run `conda init --all`.
````
`````

When it's done, open a new terminal (click on `ctrl-alt-t`) and check that the
line in the new terminal starts with `(base)`. The indication `(base)` means
that you use the base "environment".

However, working in the `base` environment is not a good practice and it is
better to disable the auto activation with:

```sh
conda config --set auto_activate_base false
```

2. Download <https://raw.githubusercontent.com/sweden-ai-factory/xai-for-life-science/refs/heads/main/environment.yaml>

:::{tip}
If you have a GPU, you might want to comment `pytorch` and uncomment `pytorch-gpu` in the 
`environment.yaml` file.
:::

3. Create an environment called `workshop`:

```shell
conda env create --yes -f environment.yaml
```

3. Activate the environment:

```shell
conda activate workshop
```

(pixi)=
### Using pixi

1. Install Pixi by following the [installation instructions](https://pixi.sh/latest/#installation).

<!--2. From an empty directory, initialize a Pixi project and add Python:

```shell
pixi init
```-->

2.  Download <https://raw.githubusercontent.com/sweden-ai-factory/xai-for-life-science/refs/heads/main/pixi.toml>, and use one the below commands

```shell
pixi shell         # If you don't have a GPU
pixi shell -e gpu  # If you have a GPU
```

<!--pixi import --feature=default environment.yaml-->
<!--pixi import --feature=default requirements.txt-->

Use either `pixi shell` to enter the environment,
or prefix commands with `pixi run`.

(notebooks)=
### Getting the Notebooks

Once you have git installed, enter the following command into your terminal:

```shell
git clone https://github.com/sweden-ai-factory/xai-for-life-science
```

The repository will be cloned to a directory named `xai-for-life-science`
which will also contain the notebooks. If you want to get the latest copy,
you update this repository by using

```shell
cd xai-for-life-science
git pull
```

### Launch JupyterLab

From within your virtual environment (using `activate` script) or conda
environment (using `conda activate workshop` / `pixi shell`) navigate to the
notebooks directory and launch JupyterLab

```shell
cd xai-for-life-science/notebooks
jupyter-lab
```

which should also open JupyterLab.

<!--
(colab-fallback)=
## Fallback option: Google colab

**Alternatively** you can use [Google colab](https://colab.research.google.com/). 

If you open a Jupyter notebook here, most of the required packages are already pre-installed. Data will be downloaded/streamed automatically. Make sure that under Runtime > Change runtime type, your "Hardware accelerator" is set to T4 GPU or equivalent (you may need to do this for each notebook).
- [Session 1 Notebook](https://colab.research.google.com/drive/1e6GhTPrmN7-rVMpQmXMj7r-RQ55Dh5Sq?usp=sharing)
- [Session 2 Notebook](https://colab.research.google.com/drive/1u-ln7tlSkF86DsICCU-YDfHlyaCt95Vb?usp=sharing)
- [Session 3 Notebook](https://colab.research.google.com/drive/1Ce4nzXpj_NHAXm4fZlI9dv1zK0HrdsGD?usp=sharing)
- [Session 4 Notebook](https://colab.research.google.com/drive/14pTu8dZJvE67CjEj5UU9lp2-29RFrr5r?usp=sharing)

-->


## Checking Your Environment

Once in, you can verify your setup by running:

```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
print("PyTorch version:", torch.__version__)
```

<!--
(datasets)=
## Downloading the required datasets

Download the datasets for this workshop [here](https://drive.google.com/file/d/1gebRwvpcPmftuhgJEAy8Jftijgbcm18K/view?usp=sharing).

Unzip these file and place them in your repository.
-->


-->

[jupyter]: http://jupyter.org/
[jupyter-install]: http://jupyter.readthedocs.io/en/latest/install.html#optional-for-experienced-python-developers-installing-jupyter-with-pip
[python]: https://python.org

[conda-forge]: https://conda-forge.org/
[miniforge]: https://conda-forge.org/download/
