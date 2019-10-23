import time
from collections import defaultdict

from knowledgerepr import fieldnetwork
from modelstore.elasticstore import StoreHandler
from ontomatch import glove_api
from inputoutput import inputoutput as io
from ontomatch.matcher_lib import MatchingType
from ontomatch.ss_api import SSAPI
from ontomatch import matcher_lib as matcherlib

# store_results(path_to_results, "best_config", combined_01)
# print("best_config...OK")


class SemProp:
    def __init__(self):
        self.store_client = StoreHandler()
        self.network = None
        self.schema_sim_index = None
        self.content_sim_index = None
        self.ontomatch_api = None
        self.matchings = None

    def add_data_model(self, path_to_serialized_model):
        self.network = fieldnetwork.deserialize_network(path_to_serialized_model)
        self.schema_sim_index = io.deserialize_object(path_to_serialized_model + 'schema_sim_index.pkl')
        self.content_sim_index = io.deserialize_object(path_to_serialized_model + 'content_sim_index.pkl')

    def add_language_model(self, path_to_sem_model):
        print("Loading language model...")
        glove_api.load_model(path_to_sem_model)
        print("Loading language model...OK")

    def init_api(self):
        if self.network is None and self.schema_sim_index is None and self.content_sim_index is None:
            print('Please add the data model first (e.g. add_data_model(\'/models/example\')')
            return

        self.ontomatch_api = SSAPI(self.network, self.store_client, self.schema_sim_index, self.content_sim_index)

    def add_ontology(self, onto_name, path_to_ontology, is_parsed=True):
        if self.ontomatch_api is None:
            print('Please init api before adding the ontology (e.g. init_api()')
            return

        self.ontomatch_api.add_krs([(onto_name, path_to_ontology)], parsed=is_parsed)

    def find_syntactic_sim(self, network, kr_handlers, sim_threshold_attr=0.5, sim_threshold_rel=0.5):
        l4_matchings = matcherlib.find_relation_class_name_matchings(network, kr_handlers,
                                                                     minhash_sim_threshold=sim_threshold_rel)
        l5_matchings = matcherlib.find_relation_class_attr_name_matching(network, kr_handlers,
                                                                         minhash_sim_threshold=sim_threshold_attr)
        return l4_matchings, l5_matchings


    def find_matchings(self, om, network, kr_handlers, store_client, sim_threshold_attr=0.5, sim_threshold_rel=0.5,
                          sem_threshold_attr=0.5, sem_threshold_rel=0.5, coh_group_threshold=0.5,
                          coh_group_size_cutoff=1, sensitivity_cancellation_signal=0.4):

            l4_matchings = matcherlib.find_relation_class_name_matchings(network, kr_handlers,
                                                                         minhash_sim_threshold=sim_threshold_rel)
            l5_matchings = matcherlib.find_relation_class_attr_name_matching(network, kr_handlers,
                                                                             minhash_sim_threshold=sim_threshold_attr)
            l42_matchings, neg_l42_matchings = matcherlib.find_relation_class_name_sem_matchings(network, kr_handlers,
                                                                                                 sem_sim_threshold=sem_threshold_rel,
                                                                                                 sensitivity_neg_signal=sensitivity_cancellation_signal)
            l52_matchings, neg_l52_matchings = matcherlib.find_relation_class_attr_name_sem_matchings(network,
                                                                                                      kr_handlers,
                                                                                                      semantic_sim_threshold=sem_threshold_attr,
                                                                                                      sensitivity_neg_signal=sensitivity_cancellation_signal)
            l6_matchings, table_groups = matcherlib.find_sem_coh_matchings(network, kr_handlers,
                                                                           sem_sim_threshold=coh_group_threshold,
                                                                           group_size_cutoff=coh_group_size_cutoff)

            l4_matchings = self.remove_negative_pairs(l4_matchings, neg_l42_matchings)
            l5_matchings = self.remove_negative_pairs(l5_matchings, neg_l52_matchings)
            l42_matchings = self.coh_group_cancellation_relation(l42_matchings, l6_matchings)
            new_l52_matchings = self.coh_group_cancellation_attribute(l52_matchings, l6_matchings)

            #Build content sim
            om.priv_build_content_sim(0.6)

            l1_matchings = []
            for kr_name, kr_handler in kr_handlers.items():
                kr_class_signatures = kr_handler.get_classes_signatures()
                l1_matchings += om.compare_content_signatures(kr_name, kr_class_signatures)

            l7_matchings = matcherlib.find_hierarchy_content_fuzzy(kr_handlers, store_client)

            # print("l1 total: " + str(len(l1_matchings)))
            # print("l4 total: " + str(len(l4_matchings)))
            # print("l42 total: " + str(len(l42_matchings)))
            # print("l5 total: " + str(len(l5_matchings)))
            # print("l52 total: " + str(len(l52_matchings)))
            # print("l7 total: " + str(len(l7_matchings)))

            return l4_matchings, l5_matchings, l42_matchings, new_l52_matchings, l1_matchings, l7_matchings

    def coh_group_cancellation_attribute(self, positive_matchings, coh_groups):
        st = time.time()
        l52_dict = defaultdict(list)
        for matching in positive_matchings:
            # adapt matching to be compared to L6
            sch, cla = matching
            sch0, sch1, sch2 = sch
            idx = ((sch0, sch1, '_'), cla)
            l52_dict[idx].append(matching)

        idx_to_remove = []
        # collect idx to remove
        for k, v in l52_dict.items():
            if k not in coh_groups:
                idx_to_remove.append(k)
        # remove the indexes and take the values as matching list
        for el in idx_to_remove:
            del l52_dict[el]
        l52_matchings = []
        for k, v in l52_dict.items():
            for el in v:
                l52_matchings.append(el)

        return l52_matchings

    def coh_group_cancellation_relation(self, positive_matchings, coh_groups):
        st = time.time()
        l42_matchings_set = set(positive_matchings)

        for m in positive_matchings:
            if m not in coh_groups and m in l42_matchings_set:
                l42_matchings_set.remove(m)

        difference = list(l42_matchings_set)
        et = time.time()
        print("Cancel time: " + str((et - st)))
        return difference

    def remove_negative_pairs(self, positive_matchings, negative_matchings):
        st = time.time()
        l4_matchings_set = set(positive_matchings)
        total_cancelled = 0

        for m in negative_matchings:
            if m in positive_matchings:
                total_cancelled += 1
                l4_matchings_set.remove(m)

        set_difference = list(l4_matchings_set)
        et = time.time()
        print("Cancel time: " + str((et - st)))
        print('Cancelled: %d pairs' % total_cancelled)
        return set_difference

    def sem_prop_pipeline(self):
        all_matchings = defaultdict(list)
        l4, l5, l42, l52, l1, l7 = self.find_matchings(
            self.ontomatch_api,
            self.ontomatch_api.network,
            self.ontomatch_api.kr_handlers,
            self.store_client,
            sim_threshold_attr=0.2,
            sim_threshold_rel=0.2,
            sem_threshold_attr=0.6,
            sem_threshold_rel=0.7,
            coh_group_threshold=0.5,
            coh_group_size_cutoff=2,
            sensitivity_cancellation_signal=0.3)

        l42 = matcherlib.summarize_matchings_to_ancestor(self.ontomatch_api, l42)
        l52 = matcherlib.summarize_matchings_to_ancestor(self.ontomatch_api, l52)

        all_matchings[MatchingType.L4_CLASSNAME_RELATIONNAME_SYN] = l4
        all_matchings[MatchingType.L5_CLASSNAME_ATTRNAME_SYN] = l5
        all_matchings[MatchingType.L42_CLASSNAME_RELATIONNAME_SEM] = l42
        all_matchings[MatchingType.L52_CLASSNAME_ATTRNAME_SEM] = l52
        all_matchings[MatchingType.L1_CLASSNAME_ATTRVALUE] = l1
        all_matchings[MatchingType.L7_CLASSNAME_ATTRNAME_FUZZY] = l7

        matchings = matcherlib.combine_matchings(all_matchings)
        self.matchings = matcherlib.summarize_matchings_to_ancestor(self.ontomatch_api, self.list_from_dict(matchings))

    def list_from_dict(self, combined):
        l = []
        for k, v in combined.items():
            matchings = v.get_matchings()
            for el in matchings:
                l.append(el)
        return l


def init_test():
    sp = SemProp()
    sp.add_data_model('test/chembl22')
    sp.add_language_model('../../models/glove.6B/glove.6B.200d.txt')
    sp.init_api()
    sp.add_ontology('efo', 'cache_onto/efo.pkl')
    return sp


def test():
    sp = init_test()
    sp.sem_prop_pipeline()

