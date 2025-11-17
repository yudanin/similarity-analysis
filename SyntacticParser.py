#!/usr/bin/env python3
"""
Sentence Similarity Analysis
"""

import spacy
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from nltk.metrics.distance import edit_distance
import requests
import re
import os

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
    print("Loaded spaCy model: en_core_web_sm\n")
except OSError:
    print("spaCy model not found.")



def build_directed_label_graph(sentence: str) -> nx.DiGraph:
    """
    Build a directed label graph of the input sentence using spaCy

    Args:
        sentence: Input sentence string

    Returns:
        NetworkX directed graph representing dependency structure
    """
    # Parse sentence with spaCy
    doc = nlp(sentence)

    # Create directed graph
    G = nx.DiGraph()

    # Add nodes (tokens) - exclude punctuation
    for token in doc:
        if not token.is_punct:
            G.add_node(token.i,
                       word=token.text,
                       lemma=token.lemma_,
                       pos=token.pos_,
                       dep=token.dep_)

    # Add edges (dependency relationships) - exclude punctuation
    for token in doc:
        if not token.is_punct and token.head != token and not token.head.is_punct:
            G.add_edge(token.head.i, token.i, relation=token.dep_)

    return G


def show_directed_label_graph(graph: nx.DiGraph, title: str = "Dependency Graph") -> None:
    """
    Visualize the directed label graph for a sentence
    Shows arrows FROM head TO dependent with labels ON arrows
    Excludes punctuation

    Args:
        graph: NetworkX directed graph OR spaCy doc OR sentence string
        title: Title for the plot
    """

    # Handle different input types
    if isinstance(graph, str):
        # If string, parse it
        doc = nlp(graph)
        tokens = [token for token in doc if not token.is_punct]
    elif hasattr(graph, '__iter__') and hasattr(next(iter(graph), None), 'text'):
        # If spaCy doc
        tokens = [token for token in graph if not token.is_punct]
    else:
        # If NetworkX graph, reconstruct from nodes
        doc = nlp(" ".join([graph.nodes[n]['word'] for n in sorted(graph.nodes())]))
        tokens = [token for token in doc if not token.is_punct]

    fig, ax = plt.subplots(figsize=(len(tokens) * 1.8 + 2, 4))

    # Position words in a line
    word_y = 0.2
    spacing = 1.5
    word_positions = {}

    for i, token in enumerate(tokens):
        x = i * spacing + 1
        word_positions[token.i] = x

        # Draw word
        ax.text(x, word_y, token.text, ha='center', va='center',
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=1", facecolor="white",
                          edgecolor="black", linewidth=1.5))

    # Find root token for the special root arrow
    root_token = None
    for token in tokens:
        if token.dep_ == 'ROOT':
            root_token = token
            break

    # Draw special ROOT arrow (from above to root word)
    if root_token:
        root_x = word_positions[root_token.i]
        # Draw arrow from above pointing down to root
        arrow = FancyArrowPatch(
            (root_x, 0.9),  # From above
            (root_x, word_y + 0.15),  # To root word
            arrowstyle='->',
            mutation_scale=15,
            linewidth=2,
            color='black'
        )
        ax.add_patch(arrow)

        # Label on the arrow
        ax.text(root_x + 0.1, 0.75, 'root', ha='center', va='center',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.15", facecolor="lightcoral",
                          edgecolor="black", linewidth=1))

    # Draw dependency arrows FROM head TO dependent (excluding punctuation)
    for token in tokens:
        if token.head != token and not token.head.is_punct:  # No punctuation arrows
            # Make sure both head and dependent are in our filtered tokens
            if token.head.i in word_positions and token.i in word_positions:
                head_x = word_positions[token.head.i]
                dep_x = word_positions[token.i]

                # Calculate arrow arc height based on distance
                distance = abs(head_x - dep_x)
                arc_height = 3 + (distance * 0.1)  # Higher arcs for longer distances

                # Midpoint for arc
                mid_x = (head_x + dep_x) / 2
                mid_y = word_y + 0.15 + arc_height

                # Draw curved arrow FROM head TO dependent
                arrow = FancyArrowPatch(
                    (head_x, word_y + 0.15),  # FROM head word
                    (dep_x, word_y + 0.15),  # TO dependent word
                    arrowstyle='->',
                    mutation_scale=15,
                    linewidth=1.8,
                    color='black',
                    connectionstyle=f"arc3,rad={arc_height / distance if distance > 0 else 0.3}"
                )
                ax.add_patch(arrow)

                # Put dependency label ON TOP of the arrow
                label_x = mid_x
                label_y = mid_y

                ax.text(label_x, label_y, token.dep_, ha='center', va='center',
                        fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue",
                                  edgecolor="black", linewidth=1))

    # Set limits and styling
    ax.set_xlim(0, len(tokens) * spacing + 1)
    ax.set_ylim(0, 1.2)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.show()


def calculate_syntactic_similarity(sentence1: str, sentence2: str) -> float:
    """
    Calculate syntactic similarity between two sentences using Levenshtein distance

    Args:
        sentence1: First sentence
        sentence2: Second sentence

    Returns:
        Similarity percentage (0-100)
    """
    print(f"\n--- SYNTACTIC SIMILARITY ANALYSIS ---")
    print(f"Sentence 1: {sentence1}")
    print(f"Sentence 2: {sentence2}")

    # Build directed label graphs using spaCy
    print("\nBuilding dependency graphs with spaCy...")
    graph1 = build_directed_label_graph(sentence1)
    graph2 = build_directed_label_graph(sentence2)

    print(f"Graph 1 - Nodes: {len(graph1.nodes())}, Edges: {len(graph1.edges())}")
    print(f"Graph 2 - Nodes: {len(graph2.nodes())}, Edges: {len(graph2.edges())}")

    # Display both graphs together
    print("\nDisplaying dependency trees...")
    show_both_graphs(graph1, graph2, sentence1, sentence2)

    # Convert graphs to structured strings for Levenshtein distance
    def graph_to_structure_string(G):
        """Convert dependency graph to structured string representation"""
        # Get dependency triples: (head_lemma, relation, dependent_lemma)
        triples = []

        # Add all dependency relationships
        for edge in sorted(G.edges()):
            head_idx, dep_idx = edge
            head_lemma = G.nodes[head_idx].get('lemma', G.nodes[head_idx]['word']).lower()
            dep_lemma = G.nodes[dep_idx].get('lemma', G.nodes[dep_idx]['word']).lower()
            relation = G.edges[edge].get('relation', 'dep')
            triples.append(f"({head_lemma}-{relation}->{dep_lemma})")

        # Add POS sequence
        pos_sequence = []
        for node in sorted(G.nodes()):
            pos = G.nodes[node].get('pos', 'X')
            pos_sequence.append(pos)

        return " ".join(triples) + " | " + "_".join(pos_sequence)

    str1 = graph_to_structure_string(graph1)
    str2 = graph_to_structure_string(graph2)

    print(f"\nStructure string 1: {str1}")
    print(f"Structure string 2: {str2}")

    # Calculate Levenshtein distance
    distance = edit_distance(str1, str2)
    max_length = max(len(str1), len(str2))

    # Convert to similarity percentage
    if max_length == 0:
        similarity_pct = 100.0
    else:
        similarity_pct = ((max_length - distance) / max_length) * 100

    print(f"\nLevenshtein distance: {distance}")
    print(f"Maximum string length: {max_length}")
    print(f"Syntactic similarity: {similarity_pct:.2f}%")

    return similarity_pct


def show_both_graphs(graph1, graph2, sentence1: str, sentence2: str) -> None:
    """
    Show both dependency graphs with correct arrow directions
    Helper function for calculate_syntactic_similarity
    """

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))
    label_font_size = 10

    def draw_correct_dependencies(graph_or_sentence, ax, sentence, title):
        # Handle input - convert to spaCy doc
        if isinstance(graph_or_sentence, str):
            doc = nlp(graph_or_sentence)
        else:
            # Reconstruct from graph or use sentence
            doc = nlp(sentence)

        # Filter out punctuation
        tokens = [token for token in doc if not token.is_punct]

        # Position words
        word_y = 0.15
        spacing = 1.3
        word_positions = {}

        for i, token in enumerate(tokens):
            x = i * spacing + 0.5
            word_positions[token.i] = x

            # Draw word
            ax.text(x, word_y, token.text, ha='center', va='center',
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              edgecolor="black", linewidth=1.2))

        # Find and draw root arrow
        root_token = None
        for token in tokens:
            if token.dep_ == 'ROOT':
                root_token = token
                break

        if root_token:
            root_x = word_positions[root_token.i]
            # Root arrow from above
            arrow = FancyArrowPatch(
                (root_x, 0.8), (root_x, word_y + 0.1),
                arrowstyle='->',
                mutation_scale=12,
                linewidth=1.5,
                color='black'
            )
            ax.add_patch(arrow)

            ax.text(root_x + 0.08, 0.65, 'root', ha='center', va='center',
                    fontsize=label_font_size, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="lightcoral",
                              edgecolor="black"))

        # Draw other dependency arrows (excluding punctuation)
        for token in tokens:
            if token.head != token and not token.head.is_punct:
                # Make sure both head and dependent are in our filtered tokens
                if token.head.i in word_positions and token.i in word_positions:
                    head_x = word_positions[token.head.i]
                    dep_x = word_positions[token.i]

                    distance = abs(head_x - dep_x)
                    arc_height = 0.25 + (distance * 0.08)

                    # Arrow FROM head TO dependent
                    arrow = FancyArrowPatch(
                        (head_x, word_y + 0.1),
                        (dep_x, word_y + 0.1),
                        arrowstyle='->',
                        mutation_scale=12,
                        linewidth=1.5,
                        color='black',
                        connectionstyle=f"arc3,rad={arc_height / distance if distance > 0 else 0.25}"
                    )
                    ax.add_patch(arrow)

                    # Label on arrow
                    mid_x = (head_x + dep_x) / 2
                    #mid_y = word_y + 0.1 + arc_height
                    mid_y = word_y + 0.3

                    ax.text(mid_x, mid_y, token.dep_, ha='center', va='center',
                            fontsize=label_font_size, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue",
                                      edgecolor="black"))

        # Styling
        ax.set_xlim(0, len(tokens) * spacing + 0.5)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{title}: '{sentence}'", fontsize=12, fontweight='bold')
        ax.axis('off')

    # Draw both sentences
    draw_correct_dependencies(graph1, ax1, sentence1, "Sentence 1")
    draw_correct_dependencies(graph2, ax2, sentence2, "Sentence 2")

    plt.tight_layout()
    plt.show()


def read_api_key(filename: str) -> str:
    """
    Read API key from a text file

    Args:
        filename: Name of the file containing the API key

    Returns:
        API key as a string

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is empty or contains invalid key
    """
    try:
        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)

        with open(file_path, 'r', encoding='utf-8') as file:
            api_key = file.read().strip()

        if not api_key:
            raise ValueError(f"API key file '{filename}' is empty")

        if not api_key.startswith('sk-ant-'):
            raise ValueError(f"Invalid API key format in '{filename}'.")

        return api_key

    except FileNotFoundError:
        raise FileNotFoundError(
            f"API key file '{filename}' not found. Please create it in the same directory as this script.")
    except Exception as e:
        raise ValueError(f"Error reading API key from '{filename}': {e}")


def calculate_similarity_claude(sentence1: str, sentence2: str) -> float:
    """
    Calculate semantic similarity using Claude API

    Args:
        sentence1: First sentence
        sentence2: Second sentence

    Returns:
        Similarity percentage (0-100)
    """

    # Read API key
    api_key = read_api_key('apikey.txt')

    print(f"\n--- CLAUDE SEMANTIC SIMILARITY ANALYSIS ---")
    print(f"Sentence 1: {sentence1}")
    print(f"Sentence 2: {sentence2}")

    prompt = f"""Analyze the semantic similarity between these two sentences. Consider their meaning, conceptual overlap, and semantic relationships.

Sentence 1: "{sentence1}"
Sentence 2: "{sentence2}"

Provide a similarity score as a percentage from 0 to 100, where:
- 0% = completely different meanings  
- 100% = identical meanings

Respond with only the numerical percentage (e.g., 75)."""

    try:
        # Note: This API call requires proper authentication
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
                "x-api-key": api_key
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            if 'content' in data and len(data['content']) > 0:
                response_text = data['content'][0]['text'].strip()

                # Extract percentage
                percentage_match = re.search(r'(\d+(?:\.\d+)?)', response_text)
                if percentage_match:
                    similarity_pct = float(percentage_match.group(1))
                    print(f"Claude similarity: {similarity_pct:.2f}%")
                    return similarity_pct
                else:
                    raise ValueError("Could not parse Claude response")
            else:
                raise ValueError("No content in API response")
        else:
            raise ValueError(f"API request failed with status {response.status_code}")

    except Exception as e:
        print(f"Error calling Claude API: {e}")
        return -1


def check_available_models():
    """Check what Claude models are available"""

    # Read API key
    api_key = read_api_key('apikey.txt')

    try:
        response = requests.get(
            "https://api.anthropic.com/v1/models",
            headers={
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
                "x-api-key": "api_key"
            }
        )
        print(f"Models endpoint status: {response.status_code}")
        if response.status_code == 200:
            print(f"Available models: {response.json()}")
        else:
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")



if __name__ == "__main__":

    #check_available_models() #if you need to change the Claude model

    print("SENTENCE SIMILARITY ANALYSIS")
    print("=" * 50)

    # Sentence pairs
    test_pairs = [
        ("The professor asked the student for help.",
         "The student asked the professor for help."),
        ("The firm paid for the project with the new government.",
         "The new government projected increased pay for the firm."),
        ("The plane crashed in the desert.",
         "The cargo plane crashed in the rocky desert near the oasis at night.")
    ]

    for i, (sent1, sent2) in enumerate(test_pairs, 1):
        print(f"\n--- PAIR {i} ---")

        # Test syntactic similarity (includes visualization)
        syntactic_sim = calculate_syntactic_similarity(sent1, sent2)

        # Test Claude similarity
        claude_sim = calculate_similarity_claude(sent1, sent2)

        print(f"\nResults for Pair {i}:")
        print(f"Syntactic similarity: {syntactic_sim:.2f}%")
        print(f"Claude similarity: {claude_sim:.2f}%")
        print(f"Difference: {abs(syntactic_sim - claude_sim):.2f}%")
