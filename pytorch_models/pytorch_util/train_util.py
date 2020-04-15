def prepare_batch(mb):
  x, y = mb[0].to(device), mb[1].to(device)
  return x, y


def eval(model, valid):
  model.eval()
  tot_val = 0.0
  crt_val = 0.0
  for i in range(len(valid)):
    x, y = prepare_batch(valid[i])
    log_prob = model(x)
    yhat = log_prob.view(-1).ge(0.5)
    crt_val += yhat.eq(y.byte()).sum().item()
    tot_val += yhat.numel()
  val_acc = crt_val / tot_val
  return val_acc
