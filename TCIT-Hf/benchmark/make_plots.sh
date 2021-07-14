####### paper ############
#python ~/bin/General_Useful_Scripts/line_plot/line_plot.py HSUB_performance.txt -x_col 2 -y_col 1 -o HSUB.pdf -markers '.' -linestyles 'None' -dims '(4.5,4)' --diagonal -x_min 0  -x_max 400 -y_min 0 -y_max 400 -x_tick_inc 100 -y_tick_inc 100 -x_tick_start 0 -y_tick_start 0 -markersizes "10" -x_label "exp" -y_label "predict" -labels "HSUB"

#python ~/bin/General_Useful_Scripts/line_plot/line_plot.py HVAP_performance.txt -x_col 2 -y_col 1 -o HVAP.pdf -markers '.' -linestyles 'None' -dims '(4.5,4)' --diagonal -x_min 0  -x_max 400 -y_min 0 -y_max 400 -x_tick_inc 100 -y_tick_inc 100 -x_tick_start 0 -y_tick_start 0 -markersizes "10" -x_label "exp" -y_label "predict" -labels "HVAP"

python ~/bin/General_Useful_Scripts/line_plot/line_plot.py Solid_compare.txt -x_col 2 -y_col 1 -o solid_phase.pdf -markers '.' -linestyles 'None' -dims '(4.5,4)' --diagonal -x_min -1700  -x_max 800 -y_min -1700 -y_max 800 -x_tick_inc 800 -y_tick_inc 800 -x_tick_start -1600 -y_tick_start -1600 -markersizes "15" -x_label "Exp" -y_label "TCIT" -labels "Hf"

python ~/bin/General_Useful_Scripts/line_plot/line_plot.py Liquid_compare.txt -x_col 2 -y_col 1 -o liquid_phase.pdf -markers '.' -linestyles 'None' -dims '(4.5,4)' --diagonal -x_min -2200  -x_max 700 -y_min -2200 -y_max 700 -x_tick_inc 700 -y_tick_inc 700 -x_tick_start -2100 -y_tick_start -2100 -markersizes "15" -x_label "Exp" -y_label "TCIT" -labels "Hf"
