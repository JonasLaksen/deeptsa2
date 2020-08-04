import sys
from src.pretty_print import  print_for_master_thesis_compact, print_for_master_thesis

if(len(sys.argv) > 2 and sys.argv[2]):
    group_size = int(sys.argv[2])
else:
    group_size = 3

if(len(sys.argv) > 3):
    val_or_test = sys.argv[3]
else:
    val_or_test = 'val'


print_folder = f'{sys.argv[1]}/*/'
print_for_master_thesis(print_folder, ['features', 'layer'], group_size=group_size, val_or_test=val_or_test)
print_for_master_thesis_compact(print_folder, ['features', 'layer'], group_size=group_size, val_or_test=val_or_test)

#print_for_master_thesis(print_folder, ['dropout', 'layer', 'loss'] )
#print_for_master_thesis_compact(print_folder, ['dropout', 'layer', 'loss'], fields_to_show=['dropout', 'layer', 'loss'], show_model=False)
