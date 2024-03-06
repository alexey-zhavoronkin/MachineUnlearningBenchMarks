def load_model(p):
    
    net = models.resnet18(pretrained=False)
    num_features = net.fc.in_features
    net.fc = nn.Linear(num_features, 8)
    net.load_state_dict(torch.load(p))
    net.cuda()
    net.eval()

    return net

def load_shadow_models():
    
    ckpt_paths = glob.glob(f'./checkpoints/shadow/*.pth')
    ckpt_paths.sort()
    
    shadow_nets = []
    for p in ckpt_paths:
        net = load_model(p)
        shadow_nets.append(net)
    return shadow_nets
