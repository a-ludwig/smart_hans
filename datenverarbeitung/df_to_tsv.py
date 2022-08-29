
from szenario1 import get_test_train


test, train = get_test_train(path = "C:/Users/peter/Nextcloud/smart_hans/AP2/Daten/auf_kopf_export", window_size = 40, df_len = 800)


print(test)
print(train)



test.to_csv('example_test.tsv', sep="\t", index= False, header= False)
train.to_csv('example_train.tsv', sep="\t", index= False, header= False)