
from scenario1 import get_scenario_1


train, test = get_scenario_1(path = "C:/Users/peter/Nextcloud/smart_hans/AP2/Daten/gesammelt")


print(test)
print(train)



test.to_csv('example_test.tsv', sep="\t", index= False, header= False)
train.to_csv('example_train.tsv', sep="\t", index= False, header= False)