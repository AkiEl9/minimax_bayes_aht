from typing import List

import numpy as np
from gymnasium.spaces import Dict, Discrete
from ray.rllib import SampleBatch
from ray.rllib.models import ModelV2
from ray.rllib.models.tf import TFModelV2
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils import override, try_import_tf
from ray.rllib.utils.typing import TensorType
from ray.rllib.policy.rnn_sequencing import add_time_dimension

tf1, tf, tfv = try_import_tf()

class LSTMMeltingPot(TFModelV2):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **custom_args):
        model_config.update(**custom_args)

        self.num_outputs = action_space.n
        self.n_scenarios = model_config.get("n_scenarios", 1)
        self.multi_values = model_config.get("multi_values", True)

        self.fcnet_size = 256
        self.lstm_cell_size = 256

        super().__init__(
            obs_space, action_space, self.num_outputs, model_config, name
        )

        self.view_requirements[SampleBatch.PREV_ACTIONS] = ViewRequirement(
            SampleBatch.ACTIONS, space=self.action_space, shift=-1
        )

        previous_action_input = tf.keras.layers.Input(shape=(1,), name="prev_actions", dtype=tf.int32)
        collective_reward_input = tf.keras.layers.Input(shape=(1,), name="collective_reward", dtype=tf.float32)
        main_input = tf.keras.layers.Input(shape=obs_space["RGB"].shape, name="main_input", dtype=tf.float32)

        action_one_hot = tf.one_hot(previous_action_input, depth=self.num_outputs, dtype=tf.float32)[:, 0]

        filters = [
            [16, [8, 8], 1],
            [32, [4, 4], 1]
        ]

        last_layer = main_input
        for i, (out_size, kernel, stride) in enumerate(filters, 1):
            last_layer = tf.keras.layers.Conv2D(
                out_size,
                kernel,
                strides=stride,
                activation="relu",
                padding="valid",
                data_format="channels_last",
                name=f"conv{i}"
            )(last_layer)

        post_cnn = tf.keras.layers.Flatten()(last_layer)
        cnn_augmented = tf.keras.layers.Concatenate(
            axis=-1,
            name="post_cnn_concat"
                                                    )([
            post_cnn, action_one_hot, collective_reward_input
        ])

        fc1 = tf.keras.layers.Dense(
            self.fcnet_size,
            name="fc1",
            activation="relu"
        )(cnn_augmented)


        state_input_h = tf.keras.layers.Input(shape=(self.lstm_cell_size,), name="h", dtype=tf.float32)
        state_input_c = tf.keras.layers.Input(shape=(self.lstm_cell_size,), name="c", dtype=tf.float32)
        seq_input = tf.keras.layers.Input(shape=(), name="seq_input", dtype=tf.int32)

        lstm_input = tf.keras.layers.Concatenate(axis=-1, name="lstm_input")(
            [fc1]
        )
        timed_lstm_input = add_time_dimension(
            padded_inputs=lstm_input, seq_lens=seq_input, framework="tf"
        )
        lstm_out, h, c = tf.keras.layers.LSTM(
            self.lstm_cell_size, return_sequences=True, return_state=True, name="lstm"
        )(
            inputs=timed_lstm_input,
            mask=tf.sequence_mask(seq_input),
            initial_state=[state_input_h, state_input_c]
        )

        action_logits = tf.keras.layers.Dense(
            self.num_outputs,
            name="action_logits",
            activation="linear",
        )(lstm_out)

        values_out = tf.keras.layers.Dense(
            self.n_scenarios if self.multi_values else 1,
            name="values_out",
            activation="linear",
            use_bias=False
        )(lstm_out)

        self.base_model = tf.keras.Model(
            [main_input, previous_action_input, collective_reward_input, seq_input, state_input_h, state_input_c],
            [action_logits, values_out, h, c])


    def forward(self, input_dict, state, seq_lens):

        main_input = tf.cast(input_dict[SampleBatch.OBS]["RGB"], tf.float32) / 255.
        collective_rewards = input_dict[SampleBatch.OBS]["COLLECTIVE_REWARD"]
        # Cut off the pseudo rewards
        collective_rewards = tf.cast(tf.clip_by_value(collective_rewards - 19., 0., 1e8) > 0., tf.float32)
        prev_action = input_dict[SampleBatch.PREV_ACTIONS]

        if self.multi_values:
            self.scenario_mask = tf.one_hot(input_dict[SampleBatch.OBS]["scenario"], depth=self.n_scenarios)

        context, self._values_out, h, c = self.base_model(
            [main_input, prev_action, collective_rewards, seq_lens] + state
        )
        return tf.reshape(context, [-1, self.num_outputs]), [h, c]

    def value_function(self):
        if self.multi_values:
            return tf.reshape(
                tf.reduce_sum(self.scenario_mask * tf.reshape(self._values_out, [-1, self.n_scenarios]), axis=-1)
                , [-1])
        else:
            return tf.reshape(
                self._values_out
                , [-1])

    def metrics(self):
        if self.multi_values:
            return {
                "scenarios": tf.reduce_mean(tf.math.argmax(self.scenario_mask))
            }
        else:
            return {}

    def get_initial_state(self) -> List[TensorType]:
        return [
            np.zeros(self.lstm_cell_size, np.float32),
            np.zeros(self.lstm_cell_size, np.float32)
        ]

class CleanupModel(TFModelV2):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **custom_args):
        model_config.update(**custom_args)

        self.num_outputs = action_space.n
        self.n_scenarios = model_config.get("n_scenarios", 1)
        self.multi_values = model_config.get("multi_values", True)

        self.fcnet_size = 64
        self.lstm_cell_size = 64

        super().__init__(
            obs_space, action_space, self.num_outputs, model_config, name
        )

        self.view_requirements[SampleBatch.PREV_ACTIONS] = ViewRequirement(
            SampleBatch.ACTIONS, space=self.action_space, shift=-1
        )

        previous_action_input = tf.keras.layers.Input(shape=(1,), name="prev_actions", dtype=tf.int32)
        collective_reward_input = tf.keras.layers.Input(shape=(1,), name="collective_reward", dtype=tf.float32)
        main_input = tf.keras.layers.Input(shape=obs_space["RGB"].shape, name="main_input", dtype=tf.float32)

        action_one_hot = tf.one_hot(previous_action_input, depth=self.num_outputs, dtype=tf.float32)[:, 0]

        filters = [
            [16, [8, 8], 1],
            [32, [4, 4], 1]
        ]

        last_layer = main_input
        for i, (out_size, kernel, stride) in enumerate(filters, 1):
            last_layer = tf.keras.layers.Conv2D(
                out_size,
                kernel,
                strides=stride,
                activation="relu",
                padding="valid",
                data_format="channels_last",
                name=f"conv{i}"
            )(last_layer)

        post_cnn = tf.keras.layers.Flatten()(last_layer)
        cnn_augmented = tf.keras.layers.Concatenate(
            axis=-1,
            name="post_cnn_concat"
                                                    )([
            post_cnn, action_one_hot, collective_reward_input
        ])

        fc1 = tf.keras.layers.Dense(
            self.fcnet_size,
            name="fc1",
            activation="relu"
        )(cnn_augmented)

        state_input_h = tf.keras.layers.Input(shape=(self.lstm_cell_size,), name="h", dtype=tf.float32)
        state_input_c = tf.keras.layers.Input(shape=(self.lstm_cell_size,), name="c", dtype=tf.float32)
        seq_input = tf.keras.layers.Input(shape=(), name="seq_input", dtype=tf.int32)

        lstm_input = tf.keras.layers.Concatenate(axis=-1, name="lstm_input")(
            [fc1]
        )
        timed_lstm_input = add_time_dimension(
            padded_inputs=lstm_input, seq_lens=seq_input, framework="tf"
        )
        lstm_out, h, c = tf.keras.layers.LSTM(
            self.lstm_cell_size, return_sequences=True, return_state=True, name="lstm"
        )(
            inputs=timed_lstm_input,
            mask=tf.sequence_mask(seq_input),
            initial_state=[state_input_h, state_input_c]
        )

        action_logits = tf.keras.layers.Dense(
            self.num_outputs,
            name="action_logits",
            activation="linear",
        )(lstm_out)

        values_out = tf.keras.layers.Dense(
            self.n_scenarios if self.multi_values else 1,
            name="values_out",
            activation="linear",
            use_bias=False
        )(lstm_out)

        self.base_model = tf.keras.Model(
            [main_input, previous_action_input, collective_reward_input, seq_input, state_input_h, state_input_c],
            [action_logits, values_out, h, c])


    def forward(self, input_dict, state, seq_lens):

        main_input = tf.cast(input_dict[SampleBatch.OBS]["RGB"], tf.float32) / 255.
        collective_rewards = input_dict[SampleBatch.OBS]["COLLECTIVE_REWARD"]
        # Cut off the pseudo rewards
        if self.model_config.get("env", "") == "clean_up":
            collective_rewards = tf.cast(tf.clip_by_value(collective_rewards - 0.5, 0., 1e8) > 0., tf.float32)
        else: # cooking
            collective_rewards = tf.cast(tf.clip_by_value(collective_rewards - 19., 0., 1e8) > 0., tf.float32)
        prev_action = input_dict[SampleBatch.PREV_ACTIONS]

        if self.multi_values:
            self.scenario_mask = tf.one_hot(input_dict[SampleBatch.OBS]["scenario"], depth=self.n_scenarios)

        context, self._values_out, h, c = self.base_model(
            [main_input, prev_action, collective_rewards, seq_lens] + state
        )
        return tf.reshape(context, [-1, self.num_outputs]), [h, c]

    def value_function(self):
        if self.multi_values:
            return tf.reshape(
                tf.reduce_sum(self.scenario_mask * tf.reshape(self._values_out, [-1, self.n_scenarios]), axis=-1)
                , [-1])
        else:
            return tf.reshape(
                self._values_out
                , [-1])

    def metrics(self):
        if self.multi_values:
            return {
                "scenarios": tf.reduce_mean(tf.math.argmax(self.scenario_mask))
            }
        else:
            return {}

    def get_initial_state(self) -> List[TensorType]:
        return [
            np.zeros(self.lstm_cell_size, np.float32),
            np.zeros(self.lstm_cell_size, np.float32)
        ]