import torch
import torch.nn as nn

class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ComplexConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x_real = x[:, :self.in_channels, :, :]
        x_imag = x[:, self.in_channels:, :, :]
        out_real = self.conv_real(x_real) - self.conv_imag(x_imag)
        out_imag = self.conv_real(x_imag) + self.conv_imag(x_real)
        return torch.cat([out_real, out_imag], dim=1)

class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0):
        super(ComplexConvTranspose2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_real = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.conv_imag = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)

    def forward(self, x):
        x_real = x[:, :self.in_channels, :, :]
        x_imag = x[:, self.in_channels:, :, :]
        out_real = self.conv_real(x_real) - self.conv_imag(x_imag)
        out_imag = self.conv_real(x_imag) + self.conv_imag(x_real)
        return torch.cat([out_real, out_imag], dim=1)

class DCCRN(nn.Module):
    def __init__(self):
        super(DCCRN, self).__init__()
        self.enc1 = ComplexConv2d(1, 16, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0))
        self.ebn1 = nn.BatchNorm2d(32)
        self.eact1 = nn.PReLU()
        
        self.enc2 = ComplexConv2d(16, 32, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0))
        self.ebn2 = nn.BatchNorm2d(64)
        self.eact2 = nn.PReLU()
        
        self.enc3 = ComplexConv2d(32, 64, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0))
        self.ebn3 = nn.BatchNorm2d(128)
        self.eact3 = nn.PReLU()
        
        self.enc4 = ComplexConv2d(64, 128, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0))
        self.ebn4 = nn.BatchNorm2d(256)
        self.eact4 = nn.PReLU()
        
        self.enc5 = ComplexConv2d(128, 128, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0))
        self.ebn5 = nn.BatchNorm2d(256)
        self.eact5 = nn.PReLU()
        
        self.enc6 = ComplexConv2d(128, 128, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0))
        self.ebn6 = nn.BatchNorm2d(256)
        self.eact6 = nn.PReLU()
        
        self.lstm = nn.LSTM(input_size=1024, hidden_size=512, num_layers=2, batch_first=True, bidirectional=False)
        self.lstm_proj = nn.Linear(512, 1024)
        
        self.dec6 = ComplexConvTranspose2d(256, 128, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0), output_padding=(1, 0))
        self.dbn6 = nn.BatchNorm2d(256)
        self.dact6 = nn.PReLU()
        
        self.dec5 = ComplexConvTranspose2d(256, 128, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0), output_padding=(1, 0))
        self.dbn5 = nn.BatchNorm2d(256)
        self.dact5 = nn.PReLU()
        
        self.dec4 = ComplexConvTranspose2d(256, 64, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0), output_padding=(1, 0))
        self.dbn4 = nn.BatchNorm2d(128)
        self.dact4 = nn.PReLU()
        
        self.dec3 = ComplexConvTranspose2d(128, 32, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0), output_padding=(1, 0))
        self.dbn3 = nn.BatchNorm2d(64)
        self.dact3 = nn.PReLU()
        
        self.dec2 = ComplexConvTranspose2d(64, 16, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0), output_padding=(1, 0))
        self.dbn2 = nn.BatchNorm2d(32)
        self.dact2 = nn.PReLU()
        
        self.dec1 = ComplexConvTranspose2d(32, 1, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0), output_padding=(1, 0))

    def forward(self, x):
        x_sliced = x[:, :, :256, :]
        e1 = self.eact1(self.ebn1(self.enc1(x_sliced)))
        e2 = self.eact2(self.ebn2(self.enc2(e1)))
        e3 = self.eact3(self.ebn3(self.enc3(e2)))
        e4 = self.eact4(self.ebn4(self.enc4(e3)))
        e5 = self.eact5(self.ebn5(self.enc5(e4)))
        e6 = self.eact6(self.ebn6(self.enc6(e5)))
        
        b, c, f, t = e6.shape
        lstm_in = e6.permute(0, 3, 1, 2).reshape(b, t, c * f)
        lstm_out, _ = self.lstm(lstm_in)
        lstm_proj = self.lstm_proj(lstm_out)
        lstm_res = lstm_proj.reshape(b, t, c, f).permute(0, 2, 3, 1)
        
        def complex_concat(d_feat, e_feat):
            cd = d_feat.shape[1] // 2
            ce = e_feat.shape[1] // 2
            dr, di = d_feat[:, :cd, :, :], d_feat[:, cd:, :, :]
            er, ei = e_feat[:, :ce, :, :], e_feat[:, ce:, :, :]
            return torch.cat([torch.cat([dr, er], dim=1), torch.cat([di, ei], dim=1)], dim=1)
            
        d6 = self.dact6(self.dbn6(self.dec6(complex_concat(lstm_res, e6))))
        d5 = self.dact5(self.dbn5(self.dec5(complex_concat(d6, e5))))
        d4 = self.dact4(self.dbn4(self.dec4(complex_concat(d5, e4))))
        d3 = self.dact3(self.dbn3(self.dec3(complex_concat(d4, e3))))
        d2 = self.dact2(self.dbn2(self.dec2(complex_concat(d3, e2))))
        d1 = torch.tanh(self.dec1(complex_concat(d2, e1)))
        
        padding = torch.zeros(b, 2, 1, t, device=d1.device)
        mask = torch.cat([d1, padding], dim=2)
        
        nr, ni = x[:, 0:1, :, :], x[:, 1:2, :, :]
        mr, mi = mask[:, 0:1, :, :], mask[:, 1:2, :, :]
        er = nr * mr - ni * mi
        ei = nr * mi + ni * mr
        return torch.cat([er, ei], dim=1)
