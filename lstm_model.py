import numpy as np

class LSTMmodel:

    def init_weights(self, input_size, hidden_size):
        Weight_x = np.random.randn(hidden_size, input_size) * 0.01
        Weight_hidden = np.random.randn(hidden_size, hidden_size) * 0.01
        b = np.zeros(hidden_size)
        return {'Weight_x': Weight_x, 'Weight_hidden': Weight_hidden, 'b': b}

    def build_all_weights(self, input_size, hidden_size):
        results = {
            'Weight_forget': self.init_weights(input_size, hidden_size),
            'Weight_input': self.init_weights(input_size, hidden_size),
            'Weight_candidate': self.init_weights(input_size, hidden_size),
            'Weight_output': self.init_weights(input_size, hidden_size),
        }
        return results

    def init_output_weights(self, hidden_size):
        return {'W_y': np.random.randn(hidden_size, 1) * 0.01, 'b_y': np.zeros(1)}

    def output_prediction(self, h_final, output_weights):
        return h_final @ output_weights['W_y'] + output_weights['b_y']

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def forget_gate(self, x_t, h_prev, weights):
        return self.sigmoid(x_t @ weights['Weight_x'].T + h_prev @ weights['Weight_hidden'].T + weights['b'])

    def input_gate(self, x_t, h_prev, weights):
        return self.sigmoid(x_t @ weights['Weight_x'].T + h_prev @ weights['Weight_hidden'].T + weights['b'])

    def output_gate(self, x_t, h_prev, weights):
        return self.sigmoid(x_t @ weights['Weight_x'].T + h_prev @ weights['Weight_hidden'].T + weights['b'])

    def candidate_gate(self, x_t, h_prev, weights):
        return self.tanh(x_t @ weights['Weight_x'].T + h_prev @ weights['Weight_hidden'].T + weights['b'])

    def compute_loss(self, predictions, targets):
        return np.mean((predictions - targets) ** 2)

    def sigmoid_deriv(self, s):
        return s * (1 - s)

    def tanh_deriv(self, t):
        return 1 - t ** 2

    def lstm_cell_backward(self, dh_next, dc_next, cache_t, weights):
        x_t = cache_t['x_t']
        h_prev = cache_t['h_prev']
        c_prev = cache_t['c_prev']
        forget_gate =   cache_t['forget_gate']
        input_gate = cache_t['input_gate']
        candidate_memory =  cache_t['candidate_memory']
        output_gate =  cache_t['output_gate']
        cell_state = cache_t['cell_state']
        
        tanh_cell_state = self.tanh(cell_state)
        doutput_gate = dh_next * tanh_cell_state
        dcell_state = dc_next + dh_next * output_gate * self.tanh_deriv(tanh_cell_state)
        dforget_gate = dcell_state * c_prev
        dinput_gate = dcell_state * candidate_memory
        dcandidate_memory = dcell_state * input_gate
        dc_prev = dcell_state * forget_gate

        df_raw = dforget_gate * self.sigmoid_deriv(forget_gate)
        di_raw = dinput_gate * self.sigmoid_deriv(input_gate)
        dc_raw = dcandidate_memory * self.tanh_deriv(candidate_memory)
        do_raw = doutput_gate * self.sigmoid_deriv(output_gate)

        gradient = {'Weight_forget': {}, 
                    'Weight_input': {}, 
                    'Weight_candidate': {}, 
                    'Weight_output': {}}
        
        gradient['Weight_forget']['Weight_x'] = df_raw.T @ x_t
        gradient['Weight_forget']['Weight_hidden'] = df_raw.T @ h_prev
        gradient['Weight_forget']['b'] = df_raw.sum(axis=0)

        gradient['Weight_input']['Weight_x'] = di_raw.T @ x_t
        gradient['Weight_input']['Weight_hidden'] = di_raw.T @ h_prev
        gradient['Weight_input']['b'] = di_raw.sum(axis=0)

        gradient['Weight_candidate']['Weight_x'] = dc_raw.T @ x_t
        gradient['Weight_candidate']['Weight_hidden'] = dc_raw.T @ h_prev
        gradient['Weight_candidate']['b'] = dc_raw.sum(axis=0)

        gradient['Weight_output']['Weight_x'] = do_raw.T @ x_t
        gradient['Weight_output']['Weight_hidden'] = do_raw.T @ h_prev
        gradient['Weight_output']['b'] = do_raw.sum(axis=0)

        dh_prev = (
            df_raw @ weights['Weight_forget']['Weight_hidden'] +
            di_raw @ weights['Weight_input']['Weight_hidden'] +
            dc_raw @ weights['Weight_candidate']['Weight_hidden'] +
            do_raw @ weights['Weight_output']['Weight_hidden']
        )
        return gradient, dh_prev, dc_prev


    def bptt(self, cache, weights, dh_last, dc_last=None):
        batch_size, hidden_size = dh_last.shape
        if dc_last is None:
            dc_last = np.zeros((batch_size, hidden_size))

        total_gradient = {
            'Weight_forget': {'Weight_x': 0, 'Weight_hidden': 0, 'b': 0},
            'Weight_input': {'Weight_x': 0, 'Weight_hidden': 0, 'b': 0},
            'Weight_candidate': {'Weight_x': 0, 'Weight_hidden': 0, 'b': 0},
            'Weight_output': {'Weight_x': 0, 'Weight_hidden': 0, 'b': 0},
        }

        dh_next = dh_last
        dc_next = dc_last

        for t in range(len(cache)-1, -1, -1):
            gradientAt_t, dh_next, dc_next = self.lstm_cell_backward(dh_next, dc_next, cache[t], weights)
            for use_gate in total_gradient:
                for param in total_gradient[use_gate]:
                    total_gradient[use_gate][param] += gradientAt_t[use_gate][param]

        return total_gradient

    def output_backward(self, h_final, y_pred, y_true, output_weights):
        d_y = 2 * (y_pred - y_true) / y_true.shape[0]
        dW_y = h_final.T @ d_y
        db_y = np.sum(d_y, axis=0)
        dh_final = d_y @ output_weights['W_y'].T

        return {'W_y': dW_y, 'b_y': db_y}, dh_final

    def update_output_weights(self, output_gradient, output_weights, learning_rate):
        output_weights['W_y'] -= learning_rate * output_gradient['W_y']
        output_weights['b_y'] -= learning_rate * output_gradient['b_y']
        return output_weights

    def backward_and_update(self, gradient, weights, learning_rate):
        updated_weights = {}
        for use_gate in weights:
            updated_weights[use_gate] = {}
            for param in weights[use_gate]:
                updated_weights[use_gate][param] = (
                    weights[use_gate][param] - learning_rate * gradient[use_gate][param]
                )
        return updated_weights

    def ensemble_predict(self, list_of_weights, list_of_output_weights, X_test, method='average'):
        all_preds = []
        batch_size = X_test.shape[1]
        for i in range(len(list_of_weights)):
            weights = list_of_weights[i]
            output_weights = list_of_output_weights[i]
            hidden_size = weights['Weight_forget']['Weight_hidden'].shape[0]
            h0 = np.zeros((batch_size, hidden_size))
            c0 = np.zeros((batch_size, hidden_size))
            result = self.lstm_forward_sequence(X_test, weights, h0, c0)
            h_final = result[1]
            y_pred = self.output_prediction(h_final, output_weights)
            all_preds.append(y_pred)

        all_preds = np.stack(all_preds)
        return np.mean(all_preds, axis=0)

    def evaluate(self, weights, output_weights, X_test, y_test):
        hidden_size = weights['Weight_forget']['Weight_hidden'].shape[0]
        total_loss = 0
        predictions = []


        for i in range(len(X_test)):
            X_batch = X_test[i]
            y_batch = y_test[i]
            y_batch = np.array(y_batch).reshape(-1, 1)
            batch_size = X_batch.shape[1]
            h0 = np.zeros((batch_size, hidden_size))
            c0 = np.zeros((batch_size, hidden_size))
            result = self.lstm_forward_sequence(X_batch, weights, h0, c0)
            h_final = result[1]
            y_pred = self.output_prediction(h_final, output_weights)
            predictions.append(y_pred)
            total_loss += self.compute_loss(y_pred, y_batch)

        avg_loss = total_loss / len(X_test)
        return avg_loss, predictions

    def cell_state_update(self, forget_gate, input_gate, candidate_memory, c_prev):
        return (forget_gate * c_prev) + (input_gate * candidate_memory)

    def hidden_state_update(self, output_gate, cell_state):
        return output_gate * self.tanh(cell_state)

    def lstm_cell_forward(self, x_t, h_prev, c_prev, weights):
        forget_gate = self.forget_gate(x_t, h_prev, weights['Weight_forget'])
        input_gate = self.input_gate(x_t, h_prev, weights['Weight_input'])
        output_gate = self.output_gate(x_t, h_prev, weights['Weight_output'])
        candidate_memory = self.candidate_gate(x_t, h_prev, weights['Weight_candidate'])

        cell_state = self.cell_state_update(forget_gate, input_gate, candidate_memory, c_prev)
        hidden_state_t = self.hidden_state_update(output_gate, cell_state)

        cache_t = {
            'x_t': x_t, 'h_prev': h_prev, 'c_prev': c_prev,
            'forget_gate': forget_gate, 'input_gate': input_gate, 'candidate_memory': candidate_memory, 'output_gate': output_gate, 'cell_state': cell_state
        }
        return hidden_state_t, cell_state, cache_t

    def lstm_forward_sequence(self, X, weights, h0, c0):
        hidden_states = []
        cache = []
        h_prev, c_prev = h0, c0

        for t in range(len(X)):
            h_prev, c_prev, cache_t = self.lstm_cell_forward(X[t], h_prev, c_prev, weights)
            hidden_states.append(h_prev)
            cache.append(cache_t)

        return hidden_states, h_prev, c_prev, cache

    def prepare_batches(self, X, y, batch_size, shuffle=True):
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)
        num_samples = X.shape[0]

        indices = np.arange(num_samples)
        if shuffle:
            np.random.shuffle(indices)

        batches = []
        num_full_batches = num_samples // batch_size
        for b in range(num_full_batches):
            batch_idx = indices[b * batch_size:(b + 1) * batch_size]
            X_batch = np.transpose(X[batch_idx], (1, 0, 2))
            y_batch = y[batch_idx]
            batches.append((X_batch, y_batch))

        return batches

    def train_model(self, X_train, y_train, hidden_size, epochs, lr,
                     batch_size=8, verbose=True):
       
        input_size = np.asarray(X_train).shape[-1]

        weights = self.build_all_weights(input_size, hidden_size)
        output_weights = self.init_output_weights(hidden_size)
        loss_history = []

        for epoch in range(epochs):
            batches = self.prepare_batches(X_train, y_train, batch_size, shuffle=True)
            epoch_loss = 0.0

            for X_batch, y_batch in batches:
                current_batch_size = X_batch.shape[1]
                h0 = np.zeros((current_batch_size, hidden_size))
                c0 = np.zeros((current_batch_size, hidden_size))

                result = self.lstm_forward_sequence(X_batch, weights, h0, c0)

                h_final = result[1]
                cache = result[3]
                y_pred = self.output_prediction(h_final, output_weights)

                loss = self.compute_loss(y_pred, y_batch)
                epoch_loss += loss

                output_gradient, dh_final = self.output_backward(h_final, y_pred, y_batch, output_weights)
                gradient = self.bptt(cache, weights, dh_final)

                weights = self.backward_and_update(gradient, weights, lr)
                output_weights = self.update_output_weights(output_gradient, output_weights, lr)

            avg_epoch_loss = epoch_loss / max(len(batches), 1)
            loss_history.append(avg_epoch_loss)

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} - MSE: {avg_epoch_loss:.4f}")

        return weights, output_weights, loss_history

    def extract_mse(self, m):
        if isinstance(m, dict):
            return float(m["loss"])
        return float(m)
 
    def compute_comparison(self, baseline_metrics, ensemble_metrics):
        baseline_mse = self.extract_mse(baseline_metrics)
        ensemble_mse = self.extract_mse(ensemble_metrics)
        baseline_rmse = np.sqrt(baseline_mse)
        ensemble_rmse = np.sqrt(ensemble_mse)
        improvement_pct = ((baseline_mse - ensemble_mse) / baseline_mse) * 100
        better_model = 'ensemble' if ensemble_mse < baseline_mse else 'baseline'
 
        return {
            'baseline_mse': baseline_mse, 'baseline_rmse': baseline_rmse,
            'ensemble_mse': ensemble_mse, 'ensemble_rmse': ensemble_rmse,
            'improvement_pct': improvement_pct, 'better_model': better_model,
        }
 
    def print_comparison(self, summary):
        print("Baseline vs Ensemble")
        print(f"Baseline  - MSE: {summary['baseline_mse']:.4f} | RMSE: {summary['baseline_rmse']:.4f} ug/m3")
        print(f"Ensemble  - MSE: {summary['ensemble_mse']:.4f} | RMSE: {summary['ensemble_rmse']:.4f} ug/m3")
        if summary['better_model'] == 'ensemble':
            print(f"Ensemble reduced MSE by {summary['improvement_pct']:.2f}% vs baseline: Supports the hypothesis.")
        else:
            print(f"Ensemble did NOT beat baseline (MSE {-summary['improvement_pct']:.2f}% worse): Does not support the hypothesis.")

if __name__ == "__main__":
    np.random.seed(0)
    num_samples, sequence_length, num_features, hidden_size = 200, 24, 10, 16

    X_fake = np.random.randn(num_samples, sequence_length, num_features)
    y_fake = np.random.randn(num_samples, 1) * 20 + 80

    model = LSTMmodel()
    weights, output_weights, history = model.train_model(
        X_fake, y_fake, hidden_size=hidden_size, epochs=3, lr=0.01, batch_size=8
    )
    block_weights, block_out_weights, block_history = model.train_model(
        X_fake[:50], y_fake[:50], hidden_size=hidden_size, epochs=3, lr=0.01, batch_size=8
    )
    model.compare_baseline_vs_ensemble(
        baseline_metrics={'loss': history[-1]},
        ensemble_metrics={'loss': block_history[-1]},
    )
    print("\nlstm_model.py ran successfully end-to-end.")
