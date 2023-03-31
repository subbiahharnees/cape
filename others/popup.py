import PySimpleGUI as sg


def popup(full, pre, gui):
    layout = [
        [sg.Button(f'{" " * 7}Full-Analysis+Plots{" " * 7}'), sg.Button(f'{" " * 3}Plot Pre-Evaluated Data{" " * 3}'),
         sg.Button(f'{" " * 3}Run model in local_host{" " * 3}')]]

    event, values = sg.Window('', layout).read(close=True)
    if event == f'{" " * 7}Full-Analysis+Plots{" " * 7}':
        full(run=True)
    elif event == f'{" " * 3}Plot Pre-Evaluated Data{" " * 3}':
        pre(run=False)
    elif event == f'{" " * 3}Run model in local_host{" " * 3}':
        gui()

        # layout = [[sg.Button(f'{" " * 7}Dataset1{" " * 7}'), sg.Button(f'{" " * 7}Dataset2{" " * 7}')]]
        # event, values = sg.Window('', layout).read(close=True)
        # if event == f'{" " * 7}Dataset1{" " * 7}':
        #     gui('dataset1')

