
from scenario1 import get_scenario_1
from scenario2 import get_scenario_2


train, test = get_scenario_2(path = "C:/Users/peter/Nextcloud/smart_hans/AP2/Daten/gesammelt")


print(test)
print(train)



test.to_csv('C:/Users/peter/Documents/Projekte/HANS/matlab/example_test.tsv', sep="\t", index= False, header= False)
train.to_csv('C:/Users/peter/Documents/Projekte/HANS/matlab/example_train.tsv', sep="\t", index= False, header= False)