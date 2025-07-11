import numpy as np

def softmax_numpy(x: np.array, temparature: float=1.0) -> np.array:
        """
        Computes a temperature-scaled softmax over the input array.

        Args:
            x (np.array): Input logits.
            temparature (float, optional): Scaling factor to control randomness. Lower values make distribution sharper.

        Returns:
            np.array: Softmax-normalized probability distribution.
        """
        x = x- np.max(x)
        x = x/temparature
        return np.exp(x)/np.sum(np.exp(x), axis=-1)
    
def top_k_probas(probas: np.array, k: int=5) -> np.array:
        """
        Selects the top-k probabilities and normalizes them.

        Args:
            probas (np.array): A 1D array of probabilities.
            k (int, optional): Number of top elements to retain.

        Returns:
            Tuple[np.array, np.array]: A tuple containing:
                - top_indices_sorted (np.array): Indices of top-k probabilities.
                - top_k_probas (np.array): Normalized top-k probability values.
        """
        probas = probas.copy()
        probas /= np.sum(probas)
        top_indices_sorted = np.argsort(-probas)[:k]
        top_k_probas = probas[top_indices_sorted]
        top_k_probas /= np.sum(top_k_probas)
        return top_indices_sorted, top_k_probas

def apply_repetition_penalty(logits: np.array, generated_ids: list, penalty: float=1.1):
        """
        Applies a repetition penalty to previously generated tokens.

        This discourages the model from repeating tokens that have already been generated.

        Args:
            logits (np.array): The raw logits for the next token.
            generated_ids (list): List of token IDs that have already been generated.
            penalty (float, optional): The penalty factor to apply (must be >= 1.0).

        Returns:
            np.array: Logits modified to penalize repeated tokens.
        """
        for token_id in set(generated_ids):
            logits[token_id] /= penalty
        return logits