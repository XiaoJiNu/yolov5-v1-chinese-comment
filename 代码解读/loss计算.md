#### compute_loss函数

在utils/utils.py中

```python
'''
yolov3中是在维度为(bs,na,h,w,c+5)的tensor中对应target的那些anchor进行标记，但是yolov5是先在build_targets函数中找到那些anchor的下标，然后再将yolo层中对应的这些输出值进行相关操作再计算loss
'''

def compute_loss(p, targets, model):  # predictions, targets, model
    '''
    p: 2x3x20x20x85, 2x3x40x40x85, 2x3x80x80x85, 3个yolo层的输出tensor
    targets: 31x6，所有图片的真实方框标签
    '''
    ft = torch.cuda.FloatTensor if p[0].is_cuda else torch.Tensor
    lcls, lbox, lobj = ft([0]), ft([0]), ft([0])
    # tbox, shape:[93x4, 135x4, 135x4]   tcls, shape:[93, 135, 135]
    # indices: a list of 3 items which contain 4 tuples whose shape are [93 135 135]，这里
    # 的indices有3个元素对应3个yolo层，每个元素里面4个tuple，对应后面的b,a,gj,gi。详见后面for循环
    # anchors: [93x2, 135x2, 135x2]
    tcls, tbox, indices, anchors = build_targets(p, targets, model)  # targets
    h = model.hyp  # hyperparameters
    red = 'mean'  # Loss reduction (sum or mean)

    # Define criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=ft([h['cls_pw']]), reduction=red)
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=ft([h['obj_pw']]), reduction=red)

    # class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
    cp, cn = smooth_BCE(eps=0.0)

    # focal loss
    g = h['fl_gamma']  # focal loss gamma
    if g > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

    # per output
    nt = 0  # targets
    for i, pi in enumerate(p):  # layer index, layer predictions
        # 这里b,a,gj,gi代表的是第i层yolo层与真实方框对应的那些anchor的下标，
        # b就是这些anchor所在的图片在这个batch中的序号
        # a就是这些anchor在3个anchor中的序号
        # gj,gi就是这个anchor所在格子左上角坐标。
        # 通过b,a,gj,gi可以定位到yolo层中分配有目标的anchor的位置以及它们所在格子左上角坐标
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        tobj = torch.zeros_like(pi[..., 0])  # target obj  1x3x64x64

        nb = b.shape[0]  # number of targets
        if nb:
            nt += nb  # cumulative targets
            # 这里提出当前yolo层中分配有目标的anchor的输出值，shape: 93x85
            # prediction subset corresponding to targets  
            ps = pi[b, a, gj, gi]  

            # GIoU
            # 与inference不同，在training的时候，Detect层没有将输出值转换为预测值，而是在计算loss的
            # 时候才将输出值转换为预测值。inference的时候则是在Detect中直接转换为预测值，并且		             # inference中计算出来bx,by会加上cx,cy并转换到输入图片的尺寸上，training的时候没有
            pxy = ps[:, :2].sigmoid() * 2. - 0.5  # shape: 93x2
            pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]  # shape: 93x2
            pbox = torch.cat((pxy, pwh), 1)  # predicted box  shape: 93x4
			# giou(prediction, target)
            giou = bbox_iou(pbox.t(), tbox[i], x1y1x2y2=False, GIoU=True)  		
            # giou loss   
            lbox += (1.0 - giou).sum() if red == 'sum' else (1.0 - giou).mean()  
            
            # Obj
            # 此时得到了与target对应的那些anchor的confidence的真实标签，这里计算confidence的真实目			# 标采用了与yolov1相似的方法，计算真实方框与预测方框的iou作为是否含有目标的优化目标，只不过			   # 这里采用的是giou。其余的anchor的confidence为0
            # 问：用预测值方框与真实方框的giou作为confidence的目标值，这种操作为什么可以学习到是否有目                 标？？？预测出来的结果很差的话giou算出出来很小，那最终学习到的confidence值不会很小					吗？？？
            tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * 				                                               giou.detach().clamp(0).type(tobj.dtype)  # giou ratio

            # Class
            # yr: ps[..., 4] predict forground or background, it is not necessary to     			 # calculate class loss if 1 class, because lobj represent it
            if model.nc > 1:  # cls loss (only if multiple classes)
                # ps是当前yolo层中与target对应的那些anchor的输出值，ps[:, 5:]取出了每个anchor的
                # 目标得分以及每个类别的概率confidence, c1,c2...,cn。t生成了和ps[:, 5:]维度一样的
                # tensor
                t = torch.full_like(ps[:, 5:], cn)  # targets 93x80
                # 此句是将每个anchor对应的目标类别在t中对应的元素的值置为cp，cp是对标签进行smooth之					# 后的值，cp=1.0 - 0.5 * eps。比如第3个anchor对应的类别为20，则t[3,20]==cp,这里				# 有多个目标，所以用数组索引赋值。这样就得到了与ps[:, 5:]对应的真实类别标签
                t[range(nb), tcls[i]] = cp
                # 由于BCEcls中默认含有sigmoid()，所以这里没有对ps[:, 5:]进行显示地sigmoid()操作
                lcls += BCEcls(ps[:, 5:], t)  # BCE

            # Append targets to text file
            # with open('targets.txt', 'a') as file:
            # [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in             					# torch.cat((txy[i], twh[i]), 1)]
		
        '''
        问：pi[..., 4] shape: 2x3x20x20，这里为什么没有对pi[..., 4](即confidence)进行sigmoid()		操作，而是直接采用yolo层的输出值？？？？？
        回答：因为BCEobj函数中默认有sigmoid函数，所以没有进行sigmoid()操作
        tobj在前面已经将与target对应的anchor的confidence设置为预测值与真实方框的giou，其它anchor
        的confidence为0，所以所有anchor有目标和没有目标的loss都计算了
        '''
        lobj += BCEobj(pi[..., 4], tobj)  # obj loss
	
    # 这里是对各个loss乘以对应的权重
    lbox *= h['giou']
    lobj *= h['obj']
    lcls *= h['cls']
    bs = tobj.shape[0]  # batch size
    if red == 'sum':
        g = 3.0  # loss gain
        lobj *= g / bs
        if nt:
            lcls *= g / nt / model.nc
            lbox *= g / nt

    loss = lbox + lobj + lcls

    # why loss * bs ???  为什么要用detach()
    return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()
```


#### build_targets()函数

在utils/utils.py中，build_targets()函数的目标是找到与target对应的那些anchor在每个yolo层输出tensor中的序号。假设yolo层的输出tensor维度是(b, na, w, h, c+5)，则每个anchor的x,y,w,h需要定位到b,na,w,h的坐标，即t

~~~python
def build_targets(p, targets, model):
    '''
    p:[2x3x20x20x85, 2x3x40x40x85, 2x3x80x80x85]，p是一个list，包含3个yolo层的输出值
    targets: 31x6，表示有31个真实目标，每个目标的标签为(image,class,x,y,w,h),image表示目			       标所在的图片在这个batch中的序号，class表示目标类别。
    '''
    # Build targets for compute_loss(), input targets(image,class,x,y,w,h) 
    det = model.module.model[-1] if type(model) in (nn.parallel.DataParallel, 												nn.parallel.DistributedDataParallel) \
        else model.model[-1]  # Detect() module
    na, nt = det.na, targets.shape[0]  # number of anchors, targets
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(6, device=targets.device)  # normalized to gridspace gain
    off = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device).float()  	   # overlap offsets
    # anchor tensor: num_anchor x num_targets 3x31, 每一列代表一个目标对应的3个anchor的序号0、	 # 1、2。有31个目标所以有31列
    at = torch.arange(na).view(na, 1).repeat(1, nt)  # anchor tensor, same as 																 # .repeat_interleave(nt)

    style = 'rect4'
    for i in range(det.nl):  # 分别对3个yolo层进行处理
        # 得到第i个yolo层的anchor的参数，第一层为[[ 3.62500,  2.81250],[ 4.87500,  6.18750], 			# [11.65625, 10.18750]]
        anchors = det.anchors[i]  
        # gain[2:] represents w,h,w,h of current feature map of current yolo layer
        # 这里是将gain张量中的第3到6个元素设置为w,h,w,h.其中w,h就是当前yolo层的特征的宽、高
        gain[2:] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

        # Match targets to anchors
        # by yr,  t: num_targets x 6 = 31x6, every row represents [image, class, x, y, w, 			h] where x,y,w,h are absolute anchor coordinates and width and height on 			  current scale of current yolo feature map
        # 这里是将每个目标的坐标参数转换成在当前yolo层的特征图的维度上的坐标，因为开始target是中的坐标		 # 对输入图片归一化的的参数，这里乘以当前特征图的宽、高，得到在当前特征图上的坐标
        a, t, offsets = [], targets * gain, 0
        # if exist object execute if
        if nt:
            # yr, r: 3x31x2, contain w h of ground truth divided by w h of anchors
            r = t[None, :, 4:6] / anchors[:, None]  # wh ratio
            '''
            temp1 = torch.max(r, 1. / r)  # shape: 3x31x2
            temp2 = torch.max(r, 1. / r).max(2)   # return tuple of value and index
            # max value of w,h ratio of three anchors between all targets
            temp3 = torch.max(r, 1. / r).max(2)[0]  
              
            # matching gt boxes anchors
            r: targets的w,h与当前yolo层的anchor的w,h的比值
            1/r: anchor的w,h与target的w,h的比值
            这里求出了当前yolo层的所有anchor与所有targets的宽、高之间的比值，同时求出了比值的倒数，找			 到每个anchor与targets宽、高的比值，如果一个anchor与一个target的宽、高比值以及target				与anchor宽、高的比值小于一定阈值的的anchor，则将这个anchor分配到对应的target,即j中对应的			 元素为true。
            问：一个target可以分配给多个anchor ？？？？ 
            回答：一个target可能有多个anchor，也可能没有anchor
            j: 3x31, represents which anchor is assigned to target, in pi yolo layer
            j的每一列表示一个目标所在格子的3个anchor中，有哪些anchor被分配到了这个目标。因为每个目标所			在的格子有3个anchor，前面通过计算这3个anchor与这个目标方框宽、高的比值以及它们的倒数，对于			小于anchor_t这个阈值的元素，j中对应位置值为true，大于等于的为false。这样就完成了所有目标			 的anchor分配
            '''
            j = torch.max(r, 1. / r).max(2)[0] < model.hyp['anchor_t']  # compare  
            # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  
            # iou(3,n) = wh_iou(anchors(3,2), gwh(n,2))
			
            '''
            # a: shape is 31, an item represents an anchor id which matches a gt box
            # 这里每个目标对应的anchor在3个anchor中的序号存放在a中，假设有n个目标，根据前面r,1/r的最			     大值是否小于阈值来决定一个目标是否会被分配anchor，如果3个anchor与这个目标的r,1/r的值都			   满足要求，则这个目标将会被分配3个anchor。所以这里a元素的个数不一定等于目标的数量，而是有				  多少个anchor被分配到这些目标就会有多少个元素。这些被分配的anchor对应的真实标签存放在t中			  的一行
              重点：anchor和目标之间是相互配对，可能有目标没有分配anchor，也有anchor没有分配目标
            # 问：如果有k个目标处于同一个格子，那这个格子的anchor并没有限定为3个，最多可能有3k个				  anchor分配到这k个目标？？？但是特征图的维度是(b,na,h,w,(c+5))，一个格子最多只有3个				  anchor,这里到底是怎么回事？？？
              问：多个目标处于同一个格子，anchor到底怎么分配的？？？
              回答：根据我在github上对作者进行的提问，发现目前对于多个目标处于同一个格子的情况，他的解					决办法就是增加anchor数量。
              	   issue链接： https://github.com/ultralytics/yolov5/issues/611
            '''
            # t: 31x6, one row represents one gt box labels [image, class, x, y, w, h]
            # t的一行对应一个分配有目标的anchor的真实标签
            a, t = at[j], t.repeat(na, 1, 1)[j]  # filter

            # overlaps
            gxy = t[:, 2:4]  # grid xy   shape of gxy: 31x2  每行表示一个目标的xy坐标
            z = torch.zeros_like(gxy)
            if style == 'rect2':
                g = 0.2  # offset
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                a, t = torch.cat((a, a[j], a[k]), 0), torch.cat((t, t[j], t[k]), 0)
                offsets = torch.cat((z, z[j] + off[0], z[k] + off[1]), 0) * g

            elif style == 'rect4':
                g = 0.5  # offset
                '''
                # (gxy % 1. < g): 以每个格子左上角为原点,水平，垂直方向为x,y轴的坐标系。判断每个目标			       方框的中心坐标x',y'(即gxy中一行元素)，在它所在的这个格子的坐标系中是否满足0<x'<0.5                   , 0<y'<0.5。
                # (gxy > 1.): 以输入图片左上角为原点的坐标系中(实质就是图片被划分的网格坐标系，尺寸也				    就是当前特征图的维度)，判断每个目标方框中心坐标x,y是否大于1
                # j：长度为31的tensor,每个元素表示每个目标方框的x坐标是否满足0<x'<0.5 且 x>1，满足                   为true,否则为false. true表示这个目标在网格坐标系中x大于1并且在目标所在格子中处于格                   子中心的左边
                # k：长度为31的tensor，每个元素表示每个目标方框的y坐标是否满足0<y'<0.5 且 y>1，满足					为true,否则为false. true表示这个目标在网格坐标系中y大于1并且在目标所在格子中处于格					子中心的上边
                # l,m同理，其中为true的元素表示这个目标是否满足0.5<x'<1, 0.5<y'<1且x坐标小于当前					  yolo层特征图的宽w-1, y坐标小于特征图的高h-1。
                # l: 长度为31的tensor,实质是x<w-1,且在目标所在格子中心的右边的目标为true，其余为
                  false
                # m: 长度为31的tensor,实质是y<h-1,且在目标所在格子中心的下边的目标为true，其余为				  false
                '''
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxy % 1. > (1 - g)) & (gxy < (gain[[2, 3]] - 1.))).T
                # 这里是将前面由j,k,l,m提取出来的目标的真实标签与原来的目标的方框真实标签堆叠起来，增					  加了目标标签数量
                a, t = torch.cat((a, a[j], a[k], a[l], a[m]), 0), torch.cat((t, t[j], 									  t[k], t[l], t[m]), 0)
                '''
                off = [[1, 0], [0, 1], [-1, 0], [0, -1]]
                offsets的计算是以当前yolo层的特征图网格为坐标系
                z是原始目标的偏移量，为0. 
                z[j]+off[0], 将处于格子中心左边的目标向右移动1
                z[k]+off[1], 将处于格子中心上边的目标向下移动1
                z[l]+off[2], 将处于格子中心右边的目标向左移动1
                z[m]+off[3], 将处于格子中心下边的目标向上移动1
                重点：对于每个分配了anchor的目标，将它们在格子中相对于格子中心的位置分为上、下、左、右
                     然后将这些目标按照这个相对位置的相反方向移动1，类似于目标抖动，增加了目标的数量
                     这么做的理由是什么？？？
                '''
                offsets = torch.cat((z, z[j] + off[0], z[k] + off[1], z[l] + off[2], z[m] 									   + off[3]), 0) * g

        # Define
        # b是一个长度为93的tensor,每个元素表示一个目标所处的图片在这个batch中的序号
        # c是一个长度为93的tensor,每个元素表示一个目标的类别
        b, c = t[:, :2].long().T  # image, class
        gxy = t[:, 2:4]  # grid xy 维度：93x2，所有目标的真实xy坐标(在当前特征图的网格坐标系下)
        gwh = t[:, 4:6]  # grid wh 维度：93x2，所有目标的真实wh值(在当前特征图的网格坐标系下)
        gij = (gxy - offsets).long()  # 所有目标所处格子的左上角坐标xy
        # gi,gj都是长度为93的一维tensor，表示每个被分配了目标的anchor所在格子的左上角坐标
        gi, gj = gij.T  # grid xy indices  
		
        # indices用于定位到被分配了目标的anchor在当前yolo层的特征图上的位置，用b,a,gj,gi可以		      唯一确定一个anchor在特征图上的位置
        # Append
        indices.append((b, a, gj, gi))  # image, anchor, grid indices
        '''
        tbox中存放的是被分配了目标的anchor对应的它真正需要学习的目标tx,ty,gw,gh,其中
        tx = gx - gi, gx是这个anchor对应的目标方框中心x坐标，gi是这个目标所在格子的左上角x坐标
        ty = gy - gj, gy是这个anchor对应的目标方框中心y坐标，gi是这个目标所在格子的左上角y坐标
        对于x,y坐标，网络需要学习的输出就是，被分配了目标的anchor所对应的目标中心距离它所在格子左上角的		偏移量tx,ty
        gwh:对于目标的宽高，网络需要学习的输出就是被分配了目标的anchor所对应的目标的宽、高w,h
        注意：这里的tx,ty,w,h的计算都是将真实标签的坐标缩放到当前yolo层的特征维度上进行计算的
        '''
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        # anchors[a]保存的是被分配了目标的anchor在当前yolo层的w,h值
        anch.append(anchors[a])  # anchors
        # tcls保存的是被分配了目标的anchor对应目标的真实类别
        tcls.append(c)  # class
        
	'''
	# tcls: indices是一个list,包含三个tensor，每个tensor的元素个数是93, 135, 135，对应3个yolo层中	   	     被分配了目标的anchor对应目标的类别
    # tbox: 包含三个元素的list，维度是[93x4, 135x4, 135x4], 93, 135, 135 表示3个yolo层各自有93，	       135，135个anchor被分配了目标，tbox中保存了这些anchor对应目标方框的tx,ty,gw,gh。这是网络需	      要学习的值
    # indices: 包含三个元素的list, 每个元素是一个元组(b,a,gj,gi),代表一个yolo层中分配有目标的anchor	         在当前yolo层中的特征图里面的位置索引。根据这些索引，可以定位到分配有目标的anchor在当前yolo层
    	  的特征网格中的位置以及所在格子的左上角坐标
          b: 分配有目标的anchor的目标所在图片在这个batch中的序号
          a: 分配有目标的anchor在3个anchor中的序号,值为0,1,2中的一个
          gj: 分配有目标的anchor在当前yolo层中的特征网格里面，所在格子的左上角y坐标
          gi: 分配有目标的anchor在当前yolo层中的特征网格里面，所在格子的左上角y坐标
	'''
    return tcls, tbox, indices, anch

~~~

