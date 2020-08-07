###yolo.py解析

流程图的的**模型构建**模块主要在yolo.py中完成，根据yaml文件实例化class Model并在Model完成初始化。
parse_model函数完成模型构建，Detect函数对各个yolo层的输入进行处理，得到训练和inference需要的预测结果。

**class Model**：用于实例化一个模型并完成模型初始化
**parse_model函数**：根据yaml文件参数创建模型，每个模块保存在一个nn.Sequential中，最终模型的所有模型都   		保存到一个nn.Sequential中
**Detect函数**：对各个yolo层的输入进行处理，得到到最终的预测结果，用于训练和inference。需要注意的是在训		练的时候Detect()并没有将各个yolo层的输入变换成预测值，而是在计算loss的时候才进行转换成预测值

~~~python
import argparse

from models.experimental import *


class Detect(nn.Module):
    def __init__(self, nc=80, anchors=()):  # detection layer
        super(Detect, self).__init__()
        self.stride = None  # strides computed during build
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        # shape(nl,1,na,1,1,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  
        self.export = False  # onnx export

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            # 将每个yolo层的输入张量的维度进行修改，维度变化如下
            # (bs,na*(5+C),w,h)->(bs,na,(C+5),w,h)->(bs,na,w,h,C+5)
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 								2).contiguous()
           
            if not self.training:  # inference
            '''
            假设网络输出的每个方框的参数是tx,ty,tw,th,confidence,c1...,cn
            每个anchor预测的方框坐标bx,by,bw,bh计算公式如下，其中cx,cy为anchor所在格子的左上角坐			标，stride为当前yolo层的特征维度相比与输入图片的缩小倍数
            这里bx,by直接将预测的方框缩放到输入图片的尺度上,bw,bh也是在输入图片上的维度吗？？？
            bx = {[2*sigmoid(tx)-0.5]+cx}*stride 
            by = {[2*sigmoid(ty)-0.5]+cy}*stride
            bw = pw*[2*sigmoid(tw)]**2
            bh = ph*[2*sigmoid(th)]**2
            对于confidence,c1...,cn，只需要进行sigmoid变化即可，即得到是否有目标以及具体类别的概率
            '''
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
				
                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) *                                 self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
				# 每个yolo层的输出的维度为(bs, anchorNum, na+5)，其中anchorNum是当前yolo层的					# anchor的总数，最终将所有yolo层的输出在第1个维度上进行concat堆叠，假设所有yolo层的				 # anchor总量为anchorNumTotal，最终的输出是(bs, anchorNumTotal, na+5)
                z.append(y.view(bs, -1, self.no))
		
        # 如果是训练阶段，只是将每个yolo层的输入张量的维度进行修改后直接返回
        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
                     # model, input channels, number of classes
    def __init__(self, model_cfg='yolov5s.yaml', ch=3, nc=None):  
        super(Model, self).__init__()
        if type(model_cfg) is dict:
            self.md = model_cfg  # model dict
        else:  # is *.yaml
            with open(model_cfg) as f:
                self.md = yaml.load(f, Loader=yaml.FullLoader)  # model dict

        # Define model
        if nc:
            self.md['nc'] = nc  # override yaml value
        self.model, self.save = parse_model(self.md, ch=[ch])  # model, savelist, ch_out
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        # 计算出网络从输入到输出的图片缩小倍数
        m.stride = torch.tensor([128 / x.shape[-2] for x in self.forward(torch.zeros(1,                                   ch, 128, 128))])  # forward
        '''
        原始anchor尺寸如下
        - [116,90, 156,198, 373,326]  # P5/32
        - [30,61, 62,45, 59,119]  # P4/16
        - [10,13, 16,30, 33,23]  # P3/8
		
		经过缩小后在特征图上的anchor维度如下
        tensor([[[ 3.62500,  2.81250], [ 4.87500,  6.18750], [11.65625, 10.18750]],
               [[ 1.87500,  3.81250],  [ 3.87500,  2.81250], [ 3.68750,  7.43750]],
               [[ 1.25000,  1.62500],  [ 2.00000,  3.75000], [ 4.12500,  2.87500]]])
        m.anchors /= m.stride.view(-1, 1, 1) 就是将各个anchor按照每个yolo层对应的特征维度与原图			特征相比缩小的尺度进行缩小，最后再计算预测的方框的尺寸就用这个缩小后的anchor的尺寸来计算
        '''
        m.anchors /= m.stride.view(-1, 1, 1)
        check_anchor_order(m)
        self.stride = m.stride

        # Init weights, biases
        torch_utils.initialize_weights(self)
        self._initialize_biases()  # only run once
        torch_utils.model_info(self)
        print('')

    def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [0.83, 0.67]  # scales
            y = []
            for i, xi in enumerate((x,
                                    torch_utils.scale_img(x.flip(3), s[0]),  # flip-lr 																					and scale
                                    torch_utils.scale_img(x, s[1]),  # scale
                                    )):
                # cv2.imwrite('img%g.jpg' % i, 255 * xi[0].numpy().transpose((1, 2, 0))					  [:, :, ::-1])
                y.append(self.forward_once(xi)[0])

            y[1][..., :4] /= s[0]  # scale
            y[1][..., 0] = img_size[1] - y[1][..., 0]  # flip lr
            y[2][..., :4] /= s[1]  # scale
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in 					 m.f]  # from earlier layers

            if profile:
                try:
                    import thop
                    o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2  # FLOPS
                except:
                    o = 0
                t = torch_utils.time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((torch_utils.time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        return x
	
    # initialize biases into Detect(), cf is class frequency
    def _initialize_biases(self, cf=None):  
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 						0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for f, s in zip(m.f, m.stride):  # 
            mi = self.model[f % m.i]
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / 						cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for f in sorted([x % m.i for x in m.f]):  # 爁rom
            b = self.model[f].bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%g Conv2d.bias:' + '%10.3g' * 6) % (f, *b[:5].mean(1).tolist(), 					   b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers...')
        for m in self.model.modules():
            if type(m) is Conv:
                m.conv = torch_utils.fuse_conv_and_bn(m.conv, m.bn)  # update conv
                m.bn = None  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        torch_utils.model_info(self)

def parse_model(md, ch):  # model_dict, input_channels(3)
    print('\n%3s%15s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 		          'arguments'))
    anchors, nc, gd, gw = md['anchors'], md['nc'], md['depth_multiple'],                                           md['width_multiple']
    na = (len(anchors[0]) // 2)  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    # from, number, module, args
    for i, (f, n, m, args) in enumerate(md['backbone'] + md['head']):  
        # 这里m代表的是一个网络模块的字符串，比如BottleneckCSP，nn.Conv2d等等。在
        # from models.experimental import *这句语句中，experimental.py中
        # from models.common import * 导入了这些网络模块的实现的类。所以在yolo.py中的全局命名空间
        # 将拥有这些类的全局变量。所以eval(m)会将这个字符串m变成与m名字相同的这个类
        m = eval(m) if isinstance(m, str) else m  # eval strings
        # args对应backbone和head参数list，每行的第四个参数是一个list，其中包含的元素有可能是字符串，
        # 当它是字符串的时候，用eval()将它转换为对应的值，和前面的eval(m)一样
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass
		
        # 这里计算的是深度方向的增益，也就是当前这个模块会重复堆叠n次
        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [nn.Conv2d, Conv, Bottleneck, SPP, DWConv, MixConv2d, Focus, ConvPlus,                    BottleneckCSP]:
            c1, c2 = ch[f], args[0]

            # Normal
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1.75  # exponential (default 2.0)
            #     e = math.log(c2 / ch[1]) / math.log(2)
            #     c2 = int(ch[1] * ex ** e)
            # if m != Focus:
            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2

            # Experimental
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1 + gw  # exponential (default 2.0)
            #     ch1 = 32  # ch[1]
            #     e = math.log(c2 / ch1) / math.log(2)  # level 1-n
            #     c2 = int(ch1 * ex ** e)
            # if m != Focus:
            #     c2 = make_divisible(c2, 8) if c2 != no else c2

            args = [c1, c2, *args[1:]]
            if m is BottleneckCSP:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
        elif m is Detect:
            # ch保存了每个模块输出的通道数，而是每个anchor的输出维度大小，即na * (nc + 5)
            # [(-1 if j == i else j - 1) for j, x in enumerate(ch) if x == no]在ch中挑出			# 所有yolo层在所有模块中的序号，j == i表示这是最后一个yolo层后面的Detect层，此时值为-1
            # 表示最后一个yolo层是Detect层的前面一层
            # 然后将三个yolo层的序号在list中的位置从后往前排列
            f = f or list(reversed([(-1 if j == i else j - 1) for j, x in enumerate(ch)                             if x == no]))
        else:
            c2 = ch[f]

        # 创建一个子模块，这个模块由backbone或者yolo head的list一行决定。
        # 这里的m由前面的m = eval(m) if isinstance(m, str) else m 变成了一个类名，这些类一般定义		  # 在models/commom.py中。所以此处m(*args)是对这些类进行了初始化，同时于再添加到    	       	       # nn.Sequential中。这些类其实和pytorch定义的类一样，所以可以添加到Sequential中。如果n大于		# 1，则会创建n个重复的模块，全部添加到Sequential中。然后这些Sequential添加到layers列表中，最  		   # 后全部添加到nn.Sequential中。
        # 要点：nn.Sequential()可以添加pytorch内置的类的实例如n.Conv2d(3, 3, 1, 1),也可以添加自定		   # 义的类的实例，把这些类同等看待即可。同时，Sequential也可以添加Sequential以及ModuleList()		 # 创建的类的实例，不同处在于自己定义的类或者ModuleList定义的类需要自己写inference函数
        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args) # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type，t是模块的名字
        np = sum([x.numel() for x in m_.parameters()])  # number params
        # attach index, 'from' index, type, number params。这里是对m_这个类添加了i,f，type,np		 # 这几个变量，以记录它的参数
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  
        print('%3s%15s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        # append to savelist。save会保存一个concat层对应的前面层在所有模块list中的序号
        # [6, 4, 17, 14, 21, 10]
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        # 前面m_是用Sequential存放的一些模块，这些模块是实例化的类，类中初始化了很多卷积，BN这些参数。		   # 当放入layers这个列表后，最终统一再传入Sequential，实现整个模型地创建
        layers.append(m_)
        ch.append(c2)  # ch保存每个模块的输出通道数量
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    device = torch_utils.select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
    # y = model(img, profile=True)

    # ONNX export
    # model.model[-1].export = True
    # torch.onnx.export(model, img, opt.cfg.replace('.yaml', '.onnx'), verbose=True, 		  opset_version=11)

    # Tensorboard
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter()
    # print("Run 'tensorboard --logdir=models/runs' to view tensorboard at      			  http://localhost:6006/")
    # tb_writer.add_graph(model.model, img)  # add model to tensorboard
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard
~~~



