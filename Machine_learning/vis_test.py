from datenverarbeitung import scenario3
from vis import vis_data


train, test = scenario3.get_scenario_3(path = "C:/Users/peter/Nextcloud/smart_hans/AP2/Daten/gesammelt", nr_taps= 1, move_window_by=-10)
vis_data.visualize(train)
