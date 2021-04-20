####### paper ############
# Exp TCIT S_gas
#python ~/bin/General_Useful_Scripts/line_plot/line_plot.py TCIT_Exp_S0.txt -x_col 2 -y_col 1 -o Exp_TCIT_S0.pdf -markers '.' -linestyles 'None' -dims '(4.5,4)' --diagonal -x_min 250  -x_max 900 -y_min 250 -y_max 900 -x_tick_inc 150 -y_tick_inc 150 -x_tick_start 250 -y_tick_start 250 -markersizes "15" -x_label "exp" -y_label "TCIT" -labels "TCIT"

# Exp TCIT Cv
#python ~/bin/General_Useful_Scripts/line_plot/line_plot.py TCIT_Exp_Cv.txt -x_col 2 -y_col 1 -o Exp_TCIT_Cv.pdf -markers '.' -linestyles 'None' -dims '(4.5,4)' --diagonal -x_min 0 -x_max 750 -y_min 0 -y_max 750 -x_tick_inc 150 -y_tick_inc 150 -x_tick_start 0 -y_tick_start 0 -markersizes "15" -x_label "exp" -y_label "TCIT" -labels "TCIT"

# G4 TCIT S_gas
python ~/bin/General_Useful_Scripts/line_plot/line_plot.py TCIT_G4_S0.txt -x_col 2 -y_col 1 -o G4_TCIT_S0.pdf -markers '.' -linestyles 'None' -dims '(4.5,4)' --diagonal -x_min 200  -x_max 675 -y_min 200 -y_max 675 -x_tick_inc 100 -y_tick_inc 100 -x_tick_start 250 -y_tick_start 250 -markersizes "20" -x_label "G4" -y_label "TCIT" -labels "TCIT"

# G4 TCIT Cv
python ~/bin/General_Useful_Scripts/line_plot/line_plot.py TCIT_G4_Cv.txt -x_col 2 -y_col 1 -o G4_TCIT_Cv.pdf -markers '.' -linestyles 'None' -dims '(4.5,4)' --diagonal -x_min 0  -x_max 300 -y_min 0 -y_max 300 -x_tick_inc 60 -y_tick_inc 60 -x_tick_start 0 -y_tick_start 0 -markersizes "20" -x_label "G4" -y_label "TCIT" -labels "TCIT"

# G4 TCIT Gf
python ~/bin/General_Useful_Scripts/line_plot/line_plot.py TCIT_G4_Gf.txt -x_col 2 -y_col 1 -o TCIT_G4_Gf.pdf -markers '.' -linestyles 'None' -dims '(4.5,4)' --diagonal -x_min -1000  -x_max 500 -y_min -1000 -y_max 500 -x_tick_inc 500 -y_tick_inc 500 -x_tick_start -1000 -y_tick_start -1000 -markersizes "15" -x_label "exp" -y_label "TCIT" -labels "TCIT"

###### G4 Exp #######
# Exp G4 S_gas
python ~/bin/General_Useful_Scripts/line_plot/line_plot.py G4_Exp_S0.txt -x_col 2 -y_col 1 -o Exp_G4_S0.pdf -markers '.' -linestyles 'None' -dims '(4.5,4)' --diagonal -x_min 150  -x_max 650 -y_min 150 -y_max 650 -x_tick_inc 100 -y_tick_inc 100 -x_tick_start 150 -y_tick_start 150 -markersizes "15" -x_label "exp" -y_label "G4" -labels "G4"

# Exp G4 Cv
python ~/bin/General_Useful_Scripts/line_plot/line_plot.py G4_Exp_Cv.txt -x_col 2 -y_col 1 -o Exp_G4_Cv.pdf -markers '.' -linestyles 'None' -dims '(4.5,4)' --diagonal -x_min 0 -x_max 300 -y_min 0 -y_max 300 -x_tick_inc 60 -y_tick_inc 60 -x_tick_start 0 -y_tick_start 0 -markersizes "15" -x_label "exp" -y_label "G4" -labels "G4"
