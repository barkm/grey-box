# grey-box

Grey-box method for modelling a reaction-advection-diffusion system.

## Background

Consider a chemical reaction in which the reactants _A_ and _B_ react to give the product _C_
according to the chemical reaction


<p align='center'>
<i>&alpha; A + &beta; B &#10230; &gamma; C</i>
</p>

where _&alpha;_, _&beta;_ and _&gamma;_ are referred to as stoichometric coefficients. 
The rate of this reaction is given by the reaction rate _R_. 

The reaction takes place in a channel in which a fluid flows from left to right.
The chemicals are then transported through the fluid by advection and diffusion. 
This can be modelled by the reaction-advection-diffusion equations

<p align='center'>
<i>
c&#775;<sub>1</sub> + w &middot; &nabla; c<sub>1</sub>
 - D &nabla; <sup>2</sup> c <sub>1</sub> = - &alpha; R(c<sub>1</sub>, c<sub>2</sub>) + g<sub>1</sub>
</i>
</p>

<p align='center'>
<i>
c&#775;<sub>2</sub> + w &middot; &nabla; c<sub>2</sub>
 - D &nabla; <sup>2</sup> c <sub>2</sub> = - &beta; R(c<sub>1</sub>, c<sub>2</sub>)  + g<sub>2</sub>
</i>
</p>

<p align='center'>
<i>
c&#775;<sub>3</sub> + w &middot; &nabla; c<sub>3</sub> 
- D &nabla; <sup>2</sup> c <sub>3</sub> = &gamma; R(c<sub>1</sub>, c<sub>2</sub>)  + g<sub>3</sub>
</i>
</p>
      

      
where _c<sub>1</sub> = [A]_, _c<sub>2</sub> = [B]_, _c<sub>3</sub> = [C]_ denote the 
concentrations of the chemicals, _w_ is an advective velocity field and _D_ is the diffusion constant.
The source terms _g<sub>1</sub>_, _g<sub>2</sub>_, _g<sub>2</sub>_ correspond to dissolving
chemicals into the fluid.

Now assume that the coefficients _&alpha;_, _&beta;_, _&gamma;_ and the reaction rate _R_ are unknown. 


The grey-box method implemented in this repository combines the reaction-advection-diffusion equations with a 
neural network to model the system. By measuring the concentrations of the chemicals in the fluid, the parameters 
of the network can be adjusted so as to model the stoichometric coefficients and the reaction rate. 

In the animation below we can see the two reactants _A_ and _B_ react in the channel to give the product _C_ downstream.
The black dots correspond to sensors through which the concentrations _c<sub>1</sub>_, 
_c<sub>2</sub>_ and _c<sub>3</sub>_ can me measured.

<p align='center'>
<img src=resources/simulation.gif/>
</p>

## Installation

The necessary dependencies can be installed with [Conda](https://docs.conda.io/en/latest/)

```bash
git clone https://github.com/barkm/grey-box
conda env create -n grey-box -f environment.yml
conda activate grey-box
```

## Instructions

### Generate data

Generate data by simulating the reaction-advection-diffusion system

```bash
python generate_data.py
```

The simulations can be plotted by running

```bash
python plot_simulation.py data/training.npz    # plot training data
python plot_simulation.py data/validation.npz  # plot validation data
python plot_simulation.py data/test.npz        # plot test data
```

### Train the model

The grey-box model can then be trained on the generated data by running

```bash
python train.py
```

Note that the training may take several hours to complete. 

The progress of the training can be inspected by running

```bash
python plot_results.py
```

To resume a training session, simply run the training script again. 

### Test the model

The grey-box model can be used to simulate the system by running

```
python test.py
```

The simulation can then be plotted by running 

```bash
python plot_simulation.py result/prediction.npz
```

and compared with the ground truth data by running

```bash
python plot_simulation.py result/prediction.npz data/test.npz
```

## References

[Barkman, Patrik. "Grey-box modelling of distributed parameter systems." (2018).](http://kth.diva-portal.org/smash/record.jsf?pid=diva2%3A1274745&dswid=6513)
