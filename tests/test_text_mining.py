import pytest
import pandas as pd
import sys
import os

# Add the parent directory to Python path so we can import text_mining module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from text_mining import (
    preprocess_text, combine_and_preprocess, remove_frequent_words, 
    create_binary_features, compute_tfidf_for_dataset, 
    get_cluster_tfidf_words, label_cluster_with_tfidf,
    preprocess_for_association_rules, find_frequent_itemsets,
    generate_association_rules, label_cluster_with_association_rules,
    integrate_text_mining_results, main_text_mining_workflow
)


def test_preprocessing():
    """Test text preprocessing functionality"""
    # Sample data
    sample_data = {
        'lat': [48.8566, 40.7128],
        'lon': [2.3522, -74.0060],
        'tags': ['paris, france, city', 'new york, usa, skyline'],
        'title': ['Eiffel Tower', 'Statue of Liberty']
    }
    df = pd.DataFrame(sample_data)
    
    # Preprocess
    df = combine_and_preprocess(df)
    
    # Verify tokens are created
    assert 'tokens' in df.columns
    assert len(df['tokens'].tolist()) == 2
    
    # Verify tokens are lists
    for tokens in df['tokens'].tolist():
        assert isinstance(tokens, list)
        assert len(tokens) > 0
    
    # Test remove frequent words
    filtered_tokens = remove_frequent_words(df['tokens'].tolist(), top_n=2)
    assert len(filtered_tokens) == 2
    
    # Test create binary features
    X, feature_names = create_binary_features(df['tokens'].tolist())
    assert X.shape[0] == 2  # 2 samples
    assert len(feature_names) > 0


def test_tfidf_labeling():
    """Test TF-IDF labeling on a sample cluster"""
    # Create sample data
    sample_data = {
        'lat': [48.8566, 40.7128, 51.5074, 48.8566, 40.7128],
        'lon': [2.3522, -74.0060, -0.1278, 2.3522, -74.0060],
        'tags': ['paris, france, city', 'new york, usa, skyline', 'london, uk, city',
                 'paris, france, tower', 'new york, usa, statue'],
        'title': ['Eiffel Tower', 'Statue of Liberty', 'Big Ben',
                  'Eiffel Tower View', 'Liberty Island']
    }
    df = pd.DataFrame(sample_data)
    
    # Preprocess
    df = combine_and_preprocess(df)
    
    # Test cluster (first two items)
    test_cluster = [0, 1]  # Image IDs
    
    # Label the cluster
    label = label_cluster_with_tfidf(df, test_cluster)
    
    # Verify label is a string
    assert isinstance(label, str)
    assert len(label) > 0
    
    # Get top words
    vectorizer, tfidf_matrix = compute_tfidf_for_dataset(df)
    top_words = get_cluster_tfidf_words(df, test_cluster, vectorizer, tfidf_matrix)
    
    # Verify top words are returned
    assert isinstance(top_words, list)


def test_association_rule_labeling():
    """Test association rule labeling on a sample cluster"""
    # Create sample data
    sample_data = {
        'lat': [48.8566, 40.7128, 51.5074, 48.8566, 40.7128, 48.8566],
        'lon': [2.3522, -74.0060, -0.1278, 2.3522, -74.0060, 2.3522],
        'tags': ['paris, france, city, tower', 'new york, usa, skyline, statue',
                 'london, uk, city, tower', 'paris, france, tower, eiffel',
                 'new york, usa, statue, liberty', 'paris, france, city, eiffel'],
        'title': ['Eiffel Tower', 'Statue of Liberty', 'Big Ben',
                  'Eiffel Tower View', 'Liberty Island', 'Paris City View']
    }
    df = pd.DataFrame(sample_data)
    
    # Preprocess
    df = combine_and_preprocess(df)
    
    # Test cluster (first three items - Paris-related)
    test_cluster = [0, 3, 5]  # Image IDs
    
    # Label the cluster using association rules
    label = label_cluster_with_association_rules(df, test_cluster)
    
    # Verify label is a string
    assert isinstance(label, str)
    
    # Test individual components
    transactions = preprocess_for_association_rules(df, test_cluster)
    assert isinstance(transactions, list)
    
    if len(transactions) > 0:
        frequent_itemsets = find_frequent_itemsets(transactions, min_support=0.1)
        assert isinstance(frequent_itemsets, pd.DataFrame)
        
        if len(frequent_itemsets) > 0:
            rules = generate_association_rules(frequent_itemsets, len(transactions), min_threshold=0.5)
            assert isinstance(rules, pd.DataFrame)


def test_complete_text_mining_workflow():
    """Test the complete text mining workflow on sample data"""
    # Create comprehensive sample data
    sample_data = {
        'lat': [48.8566, 40.7128, 51.5074, 48.8566, 40.7128, 48.8566, 51.5074, 48.8566],
        'lon': [2.3522, -74.0060, -0.1278, 2.3522, -74.0060, 2.3522, -0.1278, 2.3522],
        'tags': ['paris, france, city, tower, eiffel', 'new york, usa, skyline, statue, liberty',
                 'london, uk, city, tower, big ben', 'paris, france, tower, eiffel, landmark',
                 'new york, usa, statue, liberty, island', 'paris, france, city, eiffel, monument',
                 'london, uk, tower, big ben, clock', 'paris, france, tower, eiffel, view'],
        'title': ['Eiffel Tower', 'Statue of Liberty', 'Big Ben',
                  'Eiffel Tower Closeup', 'Liberty Island View', 'Paris City View',
                  'Big Ben Clock Tower', 'Eiffel Tower Night']
    }
    
    # Create a temporary CSV file for testing
    temp_file = 'temp_test_dataset.tsv'
    df = pd.DataFrame(sample_data)
    df.to_csv(temp_file, sep='\t', header=False, index=False)
    
    try:
        # Test cluster 1: Paris-related images (indices 0, 3, 5, 7)
        paris_cluster = [0, 3, 5, 7]
        
        result1 = main_text_mining_workflow(
            temp_file,
            paris_cluster,
            tfidf_top_n=5,
            assoc_min_support=0.3,
            assoc_min_confidence=0.6,
            max_label_words=6
        )
        
        # Verify result structure
        assert 'label' in result1
        assert 'supporting_info' in result1
        assert 'tfidf_results' in result1
        assert 'association_results' in result1
        
        # Verify label is a string
        assert isinstance(result1['label'], str)
        assert len(result1['label']) > 0
        
        # Test cluster 2: New York-related images (indices 1, 4)
        ny_cluster = [1, 4]
        
        result2 = main_text_mining_workflow(
            temp_file,
            ny_cluster,
            tfidf_top_n=4,
            assoc_min_support=0.2,
            assoc_min_confidence=0.5,
            max_label_words=5
        )
        
        # Verify result structure
        assert 'label' in result2
        assert isinstance(result2['label'], str)
        
        # Test cluster 3: London-related images (indices 2, 6)
        london_cluster = [2, 6]
        
        result3 = main_text_mining_workflow(
            temp_file,
            london_cluster,
            tfidf_top_n=3,
            assoc_min_support=0.1,
            assoc_min_confidence=0.4,
            max_label_words=4
        )
        
        # Verify result structure
        assert 'label' in result3
        assert isinstance(result3['label'], str)
        
    finally:
        # Clean up temporary file
        import os
        if os.path.exists(temp_file):
            os.remove(temp_file)