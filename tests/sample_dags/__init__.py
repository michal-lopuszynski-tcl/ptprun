from . import dag1, dag2, dag3, dag4, dag5, dag6

DAG_DATA_FACTORIES = {
    "dag1": dag1.get_dag1_data,
    "dag2": dag2.get_dag2_data,
    "dag3": dag3.get_dag3_data,
    "dag4": dag4.get_dag4_data,
    "dag5": dag5.get_dag5_data,
    "dag6": dag6.get_dag6_data,
}
