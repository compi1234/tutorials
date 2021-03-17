# NOTES ON INTERACTIVE VISUALIZATIONS and SIMPLE DASHBOARDS with JUPYTER
**Date: 02/12/2020**   
**Author: Dirk Van Compernolle**
## BACKGROUND

> The visualization space in the Python community is very dynamic,
> and I fully expect this list to be out of date as soon as it is published.
> Keep an eye out for what's coming in the future <cite> Jakevdp, Python Data Science Handbook, 2016.</cite>

Judging by the number of examples in that book that will no longer run out of the box, the above statement is an understatement
Moreover, what is true for visualization in general is even more true for widgets and interactivity and dashboarding.

The whole Jupyter notebook ecosystem boasts great claims in terms of simple, powerful and interactive computing.
Nevertheless, when going beyond the most simple types of interactivity one quickly runs into problems,
platform specific limitations, or complex code ... exactly what you did not hope for when choosing for the Jupyter ecosystem.

Some loose observations on what are the problems and the underlying causes:
- the rapidly evolving Jupyter ecosystem makes that the installation of several actors do not evolve in sync
- specific user needs are often a very good reason not to move at the speed of development of certain Jupyter components
- novel developments may make older stuff obsolete and abandoned
- the Python development community has a habit of breaking existing code with new developments; this is definitely so in the Jupyter world.
- While Jupyter provides interactivity by nature, it is interactivity for the advanced/expert user.  Served notebooks avoid the complexity of local installations but still lack an easy access mode for the more naive unexperienced users.  As a response to this need there has been a pletora of developments in dashboarding.
- *Google Colab* is still based on *IPython 5.5.0* while IPython is on the verge of moving on to IPython 8.0.0.   This implies a.o. lack of support for the matplotlib widget backend, poor support of ipywidgets and missing out on some critical changes in the Audio display() function.
- Interactivity being key implies the usage of widgets of all kinds.  However some common tools do not preserve the widgets: *nbviewer* can save the cell output to .html but it does not maintain the widgets.  Worse, if you make use of the ipywidgets Output() to visualize, then no output will be kept in *nbviewer*.

## NOTES on matplotlib backends
## *inline* vs *oustside* backends for matplotlib
There are a large number of backends available.   
These fall in two categories:
- backends that render the plots **inline** in the notebook: **inline, notebook, widget**
- backends that render the plots **outside** of the notebook: **qt5, tk**.  These backends offer typically higher flexibility for finetuning figure layouts.  (%matplotlib by itself invokes the qt5Agg backend)

Which one to choose may depend on personal choice, but even more so on the computing environment and the requirements for interactivity.
*inline-like* backends for matplotlib are backends that render everything inside (an output cell of) the notebook

##### inline
  + in most environments this is also the default
  + static backend most robust in simple plotting
  + by nature **non-interactive** 
  + less suited for exporting figures
  + LIMITATION :
    - interactive plots are not possible with the inline backend;  but the behavior can be mimicked by REDRAWING, i.e. any update of a figure corresponds to a CLEAR + REDRAW 
    - if mimicking interactivity by redrawing is acceptable depends on how heavy the application is and if it will hamper the desired flow of interactivity 
    - usability depends on how heavy the compute/plot code might be
  + REMARK: there should be a way to change (overcome) this behaviour with IPython.display.set_matplotlib_close(close=False); but it is more recommendable to use different backends if more advanced behaviour is needed

##### widget
  + matplotlib rendering is done INSIDE a jupyter *Output widget*, allowing figures + controls to be kept together in a controlled manner
  + this definitely enhances the interactive user experience
  + WARNING:
      - requires ipympl to be installed !  (that's not obvious from the name, thus confusing :) !)
      - poor backwards compatibility
      - not yet stable
  
##### notebook
  + a variant of inline but with some control over the plotting canvas --> resized, exported, ..
  + it has its own predefined output canvas for plotting
  + LIMITATIONS AND BUGS:
    - by design you can not use the 'output' ipywidget
    - is supposed to behave unpredictable when more than one notebook is open
    - seemingly not many fans in the community
    - given its poor support, should not be considered a serious alternative at this time

## Platforms

##### local
  + when **REAL-TIME** interaction is critical, then a local solution is the only relevant solution;  while response times have become very low with most served solutions, there will always be some lag
  + obvious the most flexible as you can tune your installation and setup to your needs

##### hosted notebooks
  + *Google Colab*:
    - popular, accessible, responsive,
    - lagging interactivity tools
  + *mybinder.org*:
    - more flexible than Colab as you can fully tailor your own environment
    - somewhat sluggish (sometimes really slow) in building the run-times
    - can serve voila renderings
  + *Kaggle* and many others

##### served dashboards
  + *voila*
    - via mybinder.org

## References
- https://jakevdp.github.io/PythonDataScienceHandbook/04.00-introduction-to-matplotlib.html
- https://medium.com/@1522933668924/using-matplotlib-in-jupyter-notebooks-comparing-methods-and-some-tips-python-c38e85b40ba1
- https://kapernikov.com/ipywidgets-with-matplotlib/
- https://ipywidgets.readthedocs.io/en/stable/
- https://mybinder.readthedocs.io/en/latest/
- https://voila.readthedocs.io/en/stable/
