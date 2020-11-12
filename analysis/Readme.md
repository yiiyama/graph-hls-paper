Analysis scripts
================

- paper_figure_3.py: Needs output of
  - find_event_for_display.py: Takes a preprocessed data file, runs the prediction event by event, and draws the event into a figure in a directory. Find a good-looking event from the figures, and run the same script specifying the event number + setting the PLOT at the top of the script to False.
  - dpde.py: Run the same data file and event by dpde.py to make the dP/dE dataset.
- paper_figure_4.py: Needs output of
  - keras/predict.py with `--h5-out`
  - hls4ml CSIM
  - cut_based_roc.py: Computes the performance of the cut-based (reference) classification.
