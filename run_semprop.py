from collections import defaultdict
from sys import argv

from inputoutput import inputoutput as io

from knowledgerepr import fieldnetwork
from modelstore.elasticstore import StoreHandler
from ontomatch import glove_api
from ontomatch import matcher_lib as matcherlib
from ontomatch.matcher_lib import MatchingType
from ontomatch.sem_prop_benchmarking import write_matchings_to, compute_pr_matchings, read
from ontomatch.ss_api import SSAPI


def generate_matchings(network, store_client, om, path_to_results):

    l7_matchings = matcherlib.find_hierarchy_content_fuzzy(om.kr_handlers, store_client)
    write_matchings_to(path_to_results + 'l7', l7_matchings)

    l4_matchings_01 = matcherlib.find_relation_class_name_matchings(network, om.kr_handlers,
                                                                    minhash_sim_threshold=0.1)
    write_matchings_to(path_to_results + 'l4', l4_matchings_01)

    l5_matchings_01 = matcherlib.find_relation_class_attr_name_matching(network, om.kr_handlers,
                                                                        minhash_sim_threshold=0.1)
    write_matchings_to(path_to_results + 'l5', l5_matchings_01)

    l42_matchings_05, neg_l42_matchings_02 = matcherlib.find_relation_class_name_sem_matchings(network, om.kr_handlers,
                                                                                         sem_sim_threshold=0.5,
                                                                                         negative_signal_threshold=0.1,
                                                                                         add_exact_matches=False,
                                                                                         penalize_unknown_word=True)
    write_matchings_to(path_to_results + 'l42', l42_matchings_05)
    write_matchings_to(path_to_results + 'neg_l42', neg_l42_matchings_02)

    l52_matchings_05, neg_l52_matchings_02 = matcherlib.find_relation_class_attr_name_sem_matchings(network, om.kr_handlers,
                                                                                          semantic_sim_threshold=0.5,
                                                                                          negative_signal_threshold=0.1,
                                                                                          add_exact_matches=False,
                                                                                          penalize_unknown_word=True)
    write_matchings_to(path_to_results + 'l52', l52_matchings_05)
    write_matchings_to(path_to_results + 'neg_l52', neg_l52_matchings_02)

    l6_matchings_02_1, table_groups = matcherlib.find_sem_coh_matchings(network, om.kr_handlers,
                                                                   sem_sim_threshold=0.2,
                                                                   group_size_cutoff=1)
    write_matchings_to(path_to_results + 'l6', l6_matchings_02_1)


def list_from_dict(combined):
    l = []
    for k, v in combined.items():
        matchings = v.get_matchings()
        for el in matchings:
            l.append(el)
    return l


def combine_matchings(l4, l5, l6, l42, l52, nl42, nl52, l7, ground_truth_matchings, om, cutting_ratio=0.8,
                      summary_threshold=1):
        print("Started computation ... ")
        l4_dict = dict()
        for matching in l4:
            l4_dict[matching] = 1
        total_cancelled = 0
        for m in nl42:
            if m in l4_dict:
                total_cancelled += 1
                l4.remove(m)

        l5_dict = dict()
        for matching in l5:
            l5_dict[matching] = 1
        total_cancelled = 0
        for m in nl52:
            if m in l5_dict:
                total_cancelled += 1
                l5.remove(m)

        l6_dict = dict()
        for matching in l6:
            l6_dict[matching] = 1

        # curate l42 with l6
        removed_l42 = 0
        for m in l42:
            if m not in l6_dict:
                removed_l42 += 1
                l42.remove(m)
        print("rem-l42: " + str(removed_l42))

        # curate l52 with l6
        # (('chemical', 'activity_stds_lookup', 'std_act_id'), ('efo', 'Metabolomic Profiling'))
        # (('chemical', 'activity_stds_lookup', '_'), ('efo', 'Experimental Factor'))
        removed_l52 = 0
        for m in l52:
            db, relation, attr = m[0]
            el = ((db, relation, '_'), m[1])
            if el not in l6_dict:
                removed_l52 += 1
                l52.remove(m)
        print("rem-l52: " + str(removed_l52))

        all_matchings = defaultdict(list)
        all_matchings[MatchingType.L4_CLASSNAME_RELATIONNAME_SYN] = l4
        all_matchings[MatchingType.L5_CLASSNAME_ATTRNAME_SYN] = l5
        all_matchings[MatchingType.L42_CLASSNAME_RELATIONNAME_SEM] = l42
        all_matchings[MatchingType.L52_CLASSNAME_ATTRNAME_SEM] = l52
        all_matchings[MatchingType.L7_CLASSNAME_ATTRNAME_FUZZY] = l7

        combined = matcherlib.combine_matchings(all_matchings)
        combined_list = list_from_dict(combined)

        print("StructS ... ")
        combined_sum = matcherlib.summarize_matchings_to_ancestor(om, combined_list,
                                                                  threshold_to_summarize=summary_threshold,
                                                                  summary_ratio=cutting_ratio)
        precision_sum, recall_sum = compute_pr_matchings(ground_truth_matchings, combined_sum)
        print("Precision: {}\nRecall: {}".format(precision_sum, recall_sum))

        return precision_sum, recall_sum


def combine_and_report_results(om, path_to_raw_data, path_to_ground_truth_file):

    # Getting ground truth
    with open(path_to_ground_truth_file, 'r') as gt:
        ground_truth_matchings_strings = gt.readlines()

    def parse_strings(list_of_strings):
        # format is: db %%% table %%% attr ==>> onto %%% class_name %%% list_of_matchers
        matchings = []
        for l in list_of_strings:
            tokens = l.split("==>>")
            sch = tokens[0]
            cla = tokens[1]
            sch_tokens = sch.split("%%%")
            sch_tokens = [t.strip() for t in sch_tokens]
            cla_tokens = cla.split("%%%")
            cla_tokens = [t.strip() for t in cla_tokens]
            matching_format = (((sch_tokens[0], sch_tokens[1], sch_tokens[2]), (cla_tokens[0], cla_tokens[1])))
            matchings.append(matching_format)
        return matchings

    ground_truth_matchings = parse_strings(ground_truth_matchings_strings)

    neg_l42 = read(path_to_raw_data + "neg_l42")
    neg_l52 = read(path_to_raw_data + "neg_l52")
    l6 = read(path_to_raw_data + "l6")
    l42 = read(path_to_raw_data + "l42")
    l52 = read(path_to_raw_data + "l52")
    l4 = read(path_to_raw_data + "l4")
    l5 = read(path_to_raw_data + "l5")
    l7 = read(path_to_raw_data + "l7")

    precision, recall = combine_matchings(l4, l5, l6, l42, l52, neg_l42, neg_l52, l7, ground_truth_matchings, om)
    return precision, recall


if __name__ == "__main__":
    """
    argv[1] - path to serialized model
    argv[2] - ontology name
    argv[3] - path to ontology
    argv[4] - path to semantic model
    argv[5] - path to output folder for generating the matchings
    argv[6] - path to gold standard
    
    Example: python run_semprop models/chembl22/ efo cache_onto/efo.pkl glove/glove.6B.100d.txt raw/ gold_standard
    """

    if len(argv) < 6:
        raise RuntimeError("Not enough arguments\nUsage: " +
                           "python run_semprop path_to_serialized_model onto_name path_to_ontology path_to_sem_model" +
                           "path_to_results path_to_gold_standard")

    path_to_serialized_model = argv[1]
    onto_name = argv[2]
    path_to_ontology = argv[3]
    path_to_sem_model = argv[4]
    path_to_results = argv[5]
    path_to_gold_standard = argv[6]

    # Deserialize model
    network = fieldnetwork.deserialize_network(path_to_serialized_model)
    # Create client
    store_client = StoreHandler()

    # Load glove model
    print("Loading language model...")
    glove_api.load_model(path_to_sem_model)
    print("Loading language model...OK")

    # Retrieve indexes
    schema_sim_index = io.deserialize_object(path_to_serialized_model + 'schema_sim_index.pkl')
    content_sim_index = io.deserialize_object(path_to_serialized_model + 'content_sim_index.pkl')

    # Create ontomatch api
    om = SSAPI(network, store_client, schema_sim_index, content_sim_index)
    # Load parsed ontology
    om.add_krs([(onto_name, path_to_ontology)], parsed=True)

    # # Build content sim
    om.priv_build_content_sim(0.6)

    print("Benchmarking matchers and linkers")
    generate_matchings(network, store_client, om, path_to_results)
    precision, recall = combine_and_report_results(om, path_to_results, path_to_gold_standard)
    print("F1-score: {}".format(2 * precision * recall / (precision + recall)))
