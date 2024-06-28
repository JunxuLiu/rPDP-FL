import copy
import numpy as np
import torch
from typing import List

def set_random_seed(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    
class _Model:
    """This is a helper class allowing to train a copy of a given model for
    num_updates steps by instantiating the user-provided optimizer.
    This class posesses method to retrieve current parameters set in np.ndarrays
    and to update the weights with a numpy list of the same size as the
    parameters of the model.
    """

    def __init__(
        self,
        model,
        train_dl,
        test_dl, # added by Junxu
        optimizer_class,
        lr,
        loss,
        metric,
        client_id=0,
        device="cuda:0",
        log=False,
        log_period=100,
        log_basename="local_model",
        logdir="./runs",
        seed=None,
        **kwargs
    ):
        """_summary_

        Parameters
        ----------
        model : torch.nn.Module
            _description_
        train_dl : torch.utils.data.DataLoader
            _description_
        optimizer_class : torch.optim
            A torch optimizer class that will be instantiated by calling:
            optimizer_class(self.model.parameters(), lr)
        lr : float
            The learning rate to use with th optimizer class.
        loss : torch.nn.modules.loss._loss
            an instantiated torch loss.
        num_rounds: int
            The number of communication rounds to do.
        log: bool
            Whether or not to log quantities with tensorboard. Defaults to False.
        client_id: int
            The id of the client for logging purposes. Default to 0.
        dp_target_epsilon: float
            The target epsilon for (epsilon, delta)-differential
             private guarantee. Defaults to None.
        dp_target_delta: float
            The target delta for (epsilon, delta)-differential
             private guarantee. Defaults to None.
        dp_max_grad_norm: float
            The maximum L2 norm of per-sample gradients;
             used to enforce differential privacy. Defaults to None.
        log_period: int
            The period at which to log quantities. Defaults to 100.
        log_basename: str
            The basename of the created log file if log=True. Defaults to fed_avg.
        logdir: str
            Where to create the log file. Defaults to ./runs.
        seed: int
            Seed provided to torch.Generator. Defaults to None.
        """
        self.model = copy.deepcopy(model)

        self._train_dl = train_dl
        self._test_dl = test_dl # added by Junxu 
        self._optimizer = optimizer_class(self.model.parameters(), lr)
        self._loss = copy.deepcopy(loss)
        self._metric = copy.deepcopy(metric)
        self._device = device
        self.model = self.model.to(self._device)
        self.num_batches_seen = 0
        self.log = log
        self.log_period = log_period
        self.client_id = client_id
        self.method = logdir.split('/')[-1]
        self.current_epoch = 0
        self.batch_size = None
        self.num_batches_per_epoch = None

        self._seed = seed
        set_random_seed(self._seed)
            
    def _make_private(self, privacy_engine):
        acct = copy.deepcopy(privacy_engine.accountant.accountants[self.client_id])
        if not isinstance(acct.sample_rate, float):
            self.model, self._optimizer, self._train_dl, acct = privacy_engine.make_private_with_fedrpdp(
                module=self.model,
                optimizer=self._optimizer,
                data_loader=self._train_dl,
                accountant=acct
            )
        else:
            self.model, self._optimizer, self._train_dl, acct = privacy_engine.make_private_with_feddp(
                module=self.model,
                optimizer=self._optimizer,
                data_loader=self._train_dl,
                accountant=acct
            )
        privacy_engine.accountant.accountants[self.client_id] = acct

    def _local_train(self, num_updates, privacy_accountant = None):
        """This method trains the model using the dataloader given
        for num_updates steps.
        """
        self.model = self.model.train()
        if privacy_accountant is None:
            train_loader_iter = iter(self._train_dl)
            i = 0
            while i < num_updates:
                try:
                    batch = next(train_loader_iter)
                except StopIteration:
                    train_loader_iter = iter(self._train_dl)
                    batch = next(train_loader_iter)
            
                batch = tuple(t.to(self._device) for t in batch)
                if len(batch) == 2: # for other datasets
                    logits = self.model(batch[0])
                    loss = self._loss(logits, batch[1])

                elif len(batch) == 4: # for snli dataset
                    inputs = {'input_ids':    batch[0],
                                'attention_mask': batch[1],
                                'token_type_ids': batch[2],
                                'labels':         batch[3]}
                    outputs = self.model(**inputs) # output = loss, logits, hidden_states, attentions
                    loss, logits = outputs[:2]

                loss.backward()
                self._optimizer.step()
                self._optimizer.zero_grad()
                i += 1

        else:
            train_loader_iter = iter(self._train_dl)
            current_batch_size, i = 0, 0
            while i < num_updates:
                try:
                    batch = next(train_loader_iter)
                except StopIteration:
                    train_loader_iter = iter(self._train_dl)
                    batch = next(train_loader_iter)
                current_batch_size += len(batch[-1]) 
                batch = tuple(t.to(self._device) for t in batch)
                if len(batch) == 2: # for other datasets
                    logits = self.model(batch[0])
                    loss = self._loss(logits, batch[1])

                elif len(batch) == 4: # for snli dataset
                    inputs = {'input_ids':    batch[0],
                                'attention_mask': batch[1],
                                'token_type_ids': batch[2],
                                'labels':         batch[3]}
                    outputs = self.model(**inputs) # output = loss, logits, hidden_states, attentions
                    loss, logits = outputs[:2]

                loss.backward()
                self._optimizer.step()
                self._optimizer.zero_grad()
                if len(privacy_accountant) and (i == privacy_accountant.history[-1][-1] - 1):
                    i += 1
                    current_batch_size = 0


    @torch.no_grad()
    def _get_current_params(self):
        """Returns the current weights of the pytorch model.

        Returns
        -------
        list[np.ndarray]
            A list of numpy versions of the weights.
        """
        return [
            param.cpu().detach().clone().numpy() for param in self.model.parameters()
        ]

    @torch.no_grad()
    def _update_params(self, new_params):
        """Update in place the weights of the pytorch model by adding the
        new_params list of the same size to it.

        """
        # update all the parameters
        for old_param, new_param in zip(self.model.parameters(), new_params):
            old_param.data += torch.from_numpy(new_param).to(old_param.device)


def compute_model_diff_squared_norm(model1: torch.nn.Module, model2: torch.nn.Module):
    """Compute the squared norm of the difference between two models.

    Parameters
    ----------
    model1 : torch.nn.Module
    model2 : torch.nn.Module
    """
    tensor1 = list(model1.parameters())
    tensor2 = list(model2.parameters())
    norm = sum([torch.sum((tensor1[i] - tensor2[i]) ** 2) for i in range(len(tensor1))])

    return norm

def compute_dot_product(model: torch.nn.Module, params):
    """Compute the dot prodcut between model and input parameters.

    Parameters
    ----------
    model : torch.nn.Module
    params : List containing model parameters
    """
    model_p = list(model.parameters())
    device = model_p[0].device
    dot_prod = sum([torch.sum(m * p.to(device)) for m, p in zip(model_p, params)])
    return dot_prod

def check_exchange_compliance(tensors_list, max_bytes, units="bytes"):
    """
    Check that for each round the quantities exchanged are below the dataset
    specific limit.
    Parameters
    ----------
    tensors_list: List[Union[torch.Tensor, np.ndarray]]
        The list of quantities sent by the client.
    max_bytes: int
        The number of bytes max to exchange per round per client.
    units: str
        The units in which to return the result. Default to bytes.$
    Returns
    -------
    int
        Returns the number of bits exchanged in the provided unit or raises an
        error if it went above the limit.
    """
    assert units in ["bytes", "bits", "megabytes", "gigabytes"]
    assert isinstance(tensors_list, list), "You should provide a list of tensors."
    assert all(
        [
            (isinstance(t, np.ndarray) or isinstance(t, torch.Tensor))
            for t in tensors_list
        ]
    )
    bytes_count = 0
    for t in tensors_list:
        if isinstance(t, np.ndarray):
            bytes_count += t.nbytes
        else:
            bytes_count += t.shape.numel() * torch.finfo(t.dtype).bits // 8
        if bytes_count > max_bytes:
            raise ValueError(
                f"You cannot send more than {max_bytes} bytes, this "
                f"round. You tried sending more than {bytes_count} bytes already"
            )
    if units == "bytes":
        res = bytes_count
    elif units == "bits":
        res = bytes_count * 8
    elif units == "megabytes":
        res = 1e-6 * bytes_count
    elif units == "gigabytes":
        res = 1e-9 * bytes_count
    else:
        raise NotImplementedError(f"{units} is not a possible unit")

    return res

def set_random_seed(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)