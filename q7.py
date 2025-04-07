from itertools import combinations

def create_candidate_itemsets(prev_frequent_itemsets, itemset_size):
    """Generate candidate itemsets of a specific size by joining previous frequent itemsets"""
    return set([
        itemset1.union(itemset2)
        for itemset1 in prev_frequent_itemsets
        for itemset2 in prev_frequent_itemsets
        if len(itemset1.union(itemset2)) == itemset_size
    ])


def prune_candidates_by_support(transaction_data, candidate_itemsets, min_support_threshold):
    """Filter out candidate itemsets that do not meet the minimum support"""
    support_count = {itemset: 0 for itemset in candidate_itemsets}
    for transaction in transaction_data:
        for itemset in candidate_itemsets:
            if itemset.issubset(transaction):
                support_count[itemset] += 1

    # Return only itemsets that meet or exceed the support threshold
    return {
        itemset: count
        for itemset, count in support_count.items()
        if count >= min_support_threshold
    }


def apriori_algorithm(transaction_data, min_support_threshold):
    """Run the Apriori algorithm to find all frequent itemsets in the dataset"""
    # Step 1: Find all frequent 1-itemsets
    initial_candidates = set(
        frozenset([item])
        for transaction in transaction_data
        for item in transaction
    )
    frequent_itemsets = prune_candidates_by_support(transaction_data, initial_candidates, min_support_threshold)
    all_frequent_itemsets = dict(frequent_itemsets)

    k = 2
    while frequent_itemsets:
        # Step 2: Generate next-size candidate itemsets
        new_candidates = create_candidate_itemsets(frequent_itemsets.keys(), k)
        # Step 3: Filter out candidates by support
        frequent_itemsets = prune_candidates_by_support(transaction_data, new_candidates, min_support_threshold)
        # Step 4: Add to result
        all_frequent_itemsets.update(frequent_itemsets)
        k += 1

    return all_frequent_itemsets


def generate_association_rules(frequent_itemsets, min_confidence_threshold):
    """Generate association rules from frequent itemsets"""
    rules = []
    for itemset in frequent_itemsets:
        if len(itemset) < 2:
            continue  # Skip itemsets that cannot be split into rule components

        for i in range(1, len(itemset)):
            for left_side in combinations(itemset, i):
                left_set = frozenset(left_side)
                right_set = itemset - left_set
                full_support = frequent_itemsets[itemset]
                left_support = frequent_itemsets.get(left_set, 0)

                if left_support == 0:
                    continue

                confidence = full_support / left_support
                if confidence >= min_confidence_threshold:
                    rules.append((left_set, right_set, confidence))

    return rules


# === Example Usage ===
transactions = [
    {'milk', 'bread', 'butter'},
    {'beer', 'bread'},
    {'milk', 'bread', 'beer', 'butter'},
    {'beer', 'butter'},
    {'bread', 'butter'}
]

min_support = 2
min_confidence = 0.6

frequent_itemsets = apriori_algorithm(transactions, min_support)
association_rules_list = generate_association_rules(frequent_itemsets, min_confidence)

# Output frequent itemsets
print("Frequent Itemsets:")
for itemset, support in frequent_itemsets.items():
    print(f"{set(itemset)}: support = {support}")

# Output association rules
print("\nAssociation Rules:")
for antecedent, consequent, confidence in association_rules_list:
    print(f"{set(antecedent)} -> {set(consequent)} (confidence: {confidence:.2f})")
